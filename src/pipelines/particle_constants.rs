
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct ParticleConstants {
    pub blocks_to_be_inserted_or_removed: i32,
    pub dummy1: i32,
    pub dummy2: i32,
    pub particle_stack: i32,
    pub held_bone_idx:i32,
    pub chunks_x: i32,
    pub chunks_z: i32,
    pub bones: i32,
    pub world_width:i32, // precomputed chunks_x*CHUNK_WIDTH
    pub world_depth:i32, // precomputed chunks_z*CHUNK_DEPTH
    pub world_area:i32, // world_depth * world_width
    pub total_chunks:i32, // chunks_x * chunks_z
    pub sensors:u32,
    pub world_blocks_to_update:i32,
    pub ambience_tick:u32,
    pub new_world_blocks_to_update:i32,
}
