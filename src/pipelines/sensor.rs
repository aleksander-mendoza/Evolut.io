
#[derive(Copy, Clone, Debug)]
#[repr(i32)]
pub enum SensorType{
    Movement = 1,
    Position = 2,
}
#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Sensor{
    sensor_type:SensorType,
    data0:u32,
    data1:u32,
    float_offset:u32,
}

impl Sensor{

    pub fn new_movement_sensor(particle_idx:u32,float_offset:u32)->Self{
        Self{sensor_type:SensorType::Movement,data0:particle_idx,data1:0,float_offset}
    }

    pub fn new_rotation_sensor(bone_root_particle_idx:u32,bone_endpoint_particle_idx:u32,float_offset:u32)->Self{
        Self{sensor_type:SensorType::Position,data0:bone_root_particle_idx,data1:bone_endpoint_particle_idx,float_offset}
    }

}