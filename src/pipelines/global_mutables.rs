
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct GlobalMutables {
    pub blocks_to_be_inserted_or_removed: i32,
    pub bones: u32,
    pub lidars: u32,
    pub held_bone_idx:i32,
    pub tick:i32,
    pub htm_entities:u32,
    pub ann_entities:u32,
    pub particles:u32,
}
