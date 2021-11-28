#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Muscle{
    constraint_id:u32,
    min_length:f32,
    max_length:f32,
}

impl Muscle{

    pub fn new(constraint_id:u32,
               min_length:f32,
               max_length:f32)->Self{
        Self{
            constraint_id,
            min_length,
            max_length
        }
    }

}