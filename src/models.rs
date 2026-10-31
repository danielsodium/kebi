pub struct Message {
    pub data: Vec<Data<String>>,
}

pub struct Data<T> {
    pub class: Classification,
    data: T,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Classification {
    Unclass = 0,
    Controlled = 1,
    Secret = 2,
    Password = 3,
}

impl<T: Clone> Data<T> {
    pub fn new(data: T) -> Self {
        Data {
            class: Classification::Unclass,
            data: data,
        }
    }

    pub fn get(&self) -> Option<&T> {
        match self.class {
            Classification::Unclass => Some(&self.data),
            Classification::Controlled => Some(&self.data),
            Classification::Secret => None,
            Classification::Password => None,
        }
    }

    pub fn dta(&self, target: Classification) -> Option<Data<T>> {
        // low to high
        if self.class < target {
            return Some(Data {
                class: target,
                data: self.data.clone(),
            });
        }
        return None;
    }
}

impl Message {
    pub fn new(data: Data<String>) -> Self {
        Message { data: vec![data] }
    }
    pub fn unclass(string: String) -> Self {
        Message::new(Data {
            class: Classification::Unclass,
            data: string,
        })
    }
    pub fn string(&self) -> String {
        let placeholder = "***".to_string();
        return self
            .data
            .iter()
            .map(|s| s.get().map(|s| s.as_str()).unwrap_or(&placeholder))
            .collect::<Vec<_>>()
            .join(" ");
    }
}
