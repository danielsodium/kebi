use super::llm_models::{ToolArg, ToolFunction};
use mlua::{Lua, MultiValue, RegistryKey, Result as LuaResult, Table, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

struct LoadedTool {
    index: usize,
    modified: Option<SystemTime>,
}

pub struct RefreshOutcome {
    pub changed: usize,
    pub errors: Vec<String>,
}

pub struct ToolRunner {
    lua: Lua,
    indicies: HashMap<String, usize>,
    pub tools: Vec<ToolFunction>,
    pub tool_funcs: Vec<RegistryKey>,
    loaded: HashMap<PathBuf, LoadedTool>,
}

impl ToolRunner {
    pub fn new() -> Self {
        let mut runner = ToolRunner {
            lua: Lua::new(),
            indicies: HashMap::new(),
            tools: Vec::new(),
            tool_funcs: Vec::new(),
            loaded: HashMap::new(),
        };

        if let Err(err) = runner.refresh() {
            eprintln!("Failed to load tools: {err}");
        }

        runner
    }

    pub fn refresh(&mut self) -> LuaResult<RefreshOutcome> {
        let home = env::var("HOME")
            .map_err(|_| mlua::Error::RuntimeError("HOME environment variable not set".into()))?;
        let tools_dir = PathBuf::from(home).join(".local/share/kebi/tools");
        let mut changed = 0;
        let mut errors = Vec::new();

        for entry in WalkDir::new(&tools_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("lua") {
                continue;
            }

            let modified = fs::metadata(path).and_then(|m| m.modified()).ok();
            let existing = self
                .loaded
                .get(path)
                .map(|loaded| (loaded.index, loaded.modified));

            match existing {
                Some((_, loaded_modified)) if loaded_modified == modified => {}
                Some((index, _)) => match load_tool_file(&self.lua, path) {
                    Ok((name, tool, key)) => {
                        self.replace_tool(path, index, name, tool, key, modified)?;
                        changed += 1;
                    }
                    Err(err) => {
                        let msg = format!("{}: {}", path.display(), err);
                        eprintln!("Failed to load Lua tool {msg}");
                        errors.push(msg);
                    }
                },
                None => match load_tool_file(&self.lua, path) {
                    Ok((name, tool, key)) => {
                        let index = self.tools.len();
                        self.indicies.insert(name, index);
                        self.tools.push(tool);
                        self.tool_funcs.push(key);
                        self.loaded
                            .insert(path.to_path_buf(), LoadedTool { index, modified });
                        changed += 1;
                    }
                    Err(err) => {
                        let msg = format!("{}: {}", path.display(), err);
                        eprintln!("Failed to load Lua tool {msg}");
                        errors.push(msg);
                    }
                },
            }
        }

        Ok(RefreshOutcome { changed, errors })
    }

    fn replace_tool(
        &mut self,
        path: &Path,
        index: usize,
        name: String,
        tool: ToolFunction,
        key: RegistryKey,
        modified: Option<SystemTime>,
    ) -> LuaResult<()> {
        let old_name = self.tools[index].name.clone();
        if self.indicies.get(&old_name) == Some(&index) {
            self.indicies.remove(&old_name);
        }

        self.indicies.insert(name, index);
        self.tools[index] = tool;

        let old_key = std::mem::replace(&mut self.tool_funcs[index], key);
        let _ = self.lua.remove_registry_value(old_key);
        self.loaded
            .insert(path.to_path_buf(), LoadedTool { index, modified });

        Ok(())
    }

    pub fn run(&self, name: &str, args: &HashMap<String, String>) -> LuaResult<String> {
        let index = *self
            .indicies
            .get(name)
            .ok_or_else(|| mlua::Error::RuntimeError(format!("unknown tool: {name}")))?;
        let tool = &self.tools[index];

        let func: mlua::Function = self.lua.registry_value(&self.tool_funcs[index])?;

        let mut values = MultiValue::new();
        for arg in &tool.args {
            let value = args.get(&arg.name).cloned().unwrap_or_default();
            values.push_back(Value::String(self.lua.create_string(&value)?));
        }

        let results: MultiValue = func.call(values)?;

        let mut parts = Vec::new();
        for value in results {
            if !matches!(value, Value::Nil) {
                parts.push(value.to_string()?);
            }
        }
        Ok(parts.join("\n"))
    }
}

fn load_tool_file(lua: &Lua, path: &Path) -> LuaResult<(String, ToolFunction, RegistryKey)> {
    let script = fs::read_to_string(path)
        .map_err(|err| mlua::Error::RuntimeError(format!("failed to read file: {err}")))?;

    let chunk = lua.load(&script).set_name(path.to_string_lossy());
    let plugin: Table = chunk
        .eval()
        .map_err(|err| mlua::Error::RuntimeError(format!("failed to parse: {err}")))?;

    let name: String = plugin.get("name")?;
    let description: String = plugin.get("description")?;
    let mut arguments = Vec::new();
    if let Ok(args_table) = plugin.get::<Table>("arguments") {
        for arg in args_table.sequence_values::<Table>() {
            let arg_table = arg?;
            arguments.push(ToolArg {
                name: arg_table.get("name")?,
                description: arg_table.get("description")?,
                options: None,
            });
        }
    }

    let run_func: mlua::Function = plugin.get("run")?;
    let run_func_key = lua.create_registry_value(run_func)?;

    Ok((
        name.clone(),
        ToolFunction {
            name,
            desc: description,
            args: arguments,
        },
        run_func_key,
    ))
}
