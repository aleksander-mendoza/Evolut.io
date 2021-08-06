#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct ParticleConstants {
    pub predefined_constraints: u32,
    pub collision_constraints: u32,
    pub solid_particles: u32,
    pub phantom_particles: u32,
    pub chunks_x: u32,
    pub chunks_z: u32,
}