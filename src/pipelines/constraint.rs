use crate::blocks::world_size::PARTICLE_DIAMETER;

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Constraint {
    pub particle1: u32,
    pub particle2: u32,
    pub constant_param: f32,
}

impl Constraint{
    pub fn collision(particle1:u32, particle2:u32)->Self{
        Self{particle1,particle2,constant_param:-PARTICLE_DIAMETER}
    }
    pub fn distance(particle1:u32, particle2:u32, dist:f32)->Self{
        debug_assert!(dist>0f32);
        Self{particle1,particle2,constant_param:dist}
    }
}
