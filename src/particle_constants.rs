#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct ParticleConstants {
    pub predefined_constraints: i32,
    pub collision_constraints: i32,
    pub solid_particles: i32,
    pub phantom_particles: i32,
    pub chunks_x: i32,
    pub chunks_z: i32,
    pub bones: i32,
    pub dummy: i32,
}