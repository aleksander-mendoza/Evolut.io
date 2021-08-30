use crate::blocks::world_size::PARTICLE_DIAMETER;

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Constraint {
    pub constant_param: glm::Vec3,
    pub constant_param1: f32,
    pub constant_param2: f32,
    pub stiffness: f32,
    pub bone_idx1: u32,
    pub bone_idx2: u32,

}

// impl Constraint{
//     // pub fn collision(particle1:u32, particle2:u32)->Self{
//     //     Self{particle1,particle2,constant_param:-PARTICLE_DIAMETER}
//     // }
//     pub fn distance(stiffness:f32, particle1:u32, particle2:u32, dist:f32)->Self{
//         debug_assert!(dist>0f32);
//         Self{stiffness,particle1,particle2,constant_param:dist}
//     }
// }
