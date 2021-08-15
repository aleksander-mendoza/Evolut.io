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
    pub world_width:i32, // precomputed chunks_x*CHUNK_WIDTH
    pub world_depth:i32, // precomputed chunks_z*CHUNK_DEPTH
    pub world_area:i32, // world_depth * world_width
    pub total_chunks:i32, // chunks_x * chunks_z
    pub dummy1:i32,
}
