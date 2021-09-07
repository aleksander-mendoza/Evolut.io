
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct GlobalMutables {
    pub blocks_to_be_inserted_or_removed: i32,
    pub bones: u32,
    pub world_blocks_to_update_even: u32,
    pub dummy2: i32,
    pub held_bone_idx:i32,
    pub ambience_tick:i32,
    pub world_blocks_to_update:[u32;2],
}
