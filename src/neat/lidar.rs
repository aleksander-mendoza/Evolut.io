#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, packed)]
pub struct Lidar {
    direction:glm::Vec3,
    prev_block_idx:u32,
    hit_block_idx:u32,
    hit_block_id:u32,
    hit_entity_id:u32,
    parent_entity_id:u32,
}
