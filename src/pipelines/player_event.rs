use crate::blocks::Block;

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum EventType{
    Nothing = 0,
    Throw = 1,
    SetBlock = 2,
}
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct ThrowData{
    position: glm::Vec3,
    dummy1:u32, // GLSL expects vec3 to be aligned to 16 bytes
    velocity: glm::Vec3,
    dummy2:u32,
}
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct SetBlockData{
    block_idx: u32,
    block: Block,
}
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub union EventData{
    throw: ThrowData,
    set_block: SetBlockData,
}
#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct PlayerEvent {
    event_type: EventType,
    data: EventData,
}