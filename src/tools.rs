use super::llm_models::{ToolArg, ToolFunction};
use mlua::{Lua, MultiValue, RegistryKey, Result as LuaResult, Table, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use walkdir::WalkDir;

pub struct ToolRunner {
    lua: Lua,
    indicies: HashMap<String, usize>,
    pub tools: Vec<ToolFunction>,
    pub tool_funcs: Vec<RegistryKey>,
}

impl ToolRunner {
    pub fn new() -> Self {
        let lua = Lua::new();

        let (indicies, tools, tool_funcs) =
            load_tools(&lua).unwrap_or_else(|_| (HashMap::new(), Vec::new(), Vec::new()));
        ToolRunner {
            lua,
            indicies,
            tools,
            tool_funcs,
        }
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

pub fn load_tools(
    lua: &Lua,
) -> LuaResult<(HashMap<String, usize>, Vec<ToolFunction>, Vec<RegistryKey>)> {
    let mut indicies = HashMap::new();
    let mut loaded_tools = Vec::new();
    let mut registries = Vec::new();
    let home = env::var("HOME").expect("HOME environment variable not set");
    let tools_dir = format!("{}/.local/share/kebi/tools", home);

    for entry in WalkDir::new(&tools_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("lua") {
            let script = fs::read_to_string(path)
                .unwrap_or_else(|_| panic!("Failed to read file: {}", path.display()));

            let chunk = lua.load(&script).set_name(path.to_string_lossy());
            let plugin: Table = match chunk.eval() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to parse Lua tool {}: {}", path.display(), e);
                    continue;
                }
            };
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

            indicies.insert(name.clone(), loaded_tools.len());
            registries.push(run_func_key);
            loaded_tools.push(ToolFunction {
                name,
                desc: description,
                args: arguments,
            });
        }
    }
    Ok((indicies, loaded_tools, registries))
}
