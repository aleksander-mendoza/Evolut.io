

pub struct SpecializationConstants{
    entries:Vec<ash::vk::SpecializationMapEntry>,
    data:Vec<u8>,
}

impl SpecializationConstants{

    pub fn new()->Self{
        Self{ entries: vec![], data: vec![] }
    }
    pub fn entry_float(&mut self, id:u32, val:f32){
        self.entry(id,&val)
    }
    pub fn entry_uint(&mut self, id:u32, val:u32){
        self.entry(id,&val)
    }
    pub fn entry<T:Copy>(&mut self, id:u32, val:&T){
        assert_eq!(self.entries.iter().find(|x|x.constant_id==id).map(|x|x.constant_id),None);
        let size = std::mem::size_of_val(val);
        self.entries.push(ash::vk::SpecializationMapEntry{
            constant_id: id,
            offset: self.data.len() as u32,
            size
        });
        let raw = unsafe{std::slice::from_raw_parts(val as *const T as *const u8,size)};
        self.data.extend_from_slice(raw);
    }

    pub fn build(&self)->ash::vk::SpecializationInfoBuilder{
        ash::vk::SpecializationInfo::builder()
            .map_entries(&self.entries)
            .data(&self.data)
    }
}