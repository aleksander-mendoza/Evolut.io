
use crate::blocks::{Block, WorldSize, BlockId};
use std::fmt::{Debug, Formatter};
use crate::blocks::block_properties::AIR;

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum EventType{
    Nothing = 0,
    Throw = 1,
    SetBlock = 2,
}
#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct PlayerEvent {
    vec3_slot0: glm::Vec3,
    u32_slot0: u32,
    vec3_slot1: glm::Vec3,
    u32_slot1: u32,
    uvec3_slot1: glm::UVec3,
    event_type: EventType,
}
impl Debug for PlayerEvent{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.event_type{
            EventType::Nothing => {
                write!(f,"Nothing")
            }
            EventType::Throw => {
                let a = self.vec3_slot0;
                let b = self.vec3_slot1;
                write!(f,"Throw{{position={}, velocity={}}}",a, b)
            }
            EventType::SetBlock => {
                let a = self.u32_slot0;
                let b = self.u32_slot1;
                write!(f,"SetBlock{{block_idx={}, block_id={}}}",a,b)
            }
        }
    }
}
impl PlayerEvent{
    pub fn nothing()->Self{
        Self{
            event_type: EventType::Nothing,
            vec3_slot0: Default::default(),
            u32_slot0: 0,
            vec3_slot1: Default::default(),
            u32_slot1: 0,
            uvec3_slot1: Default::default()
        }
    }
    pub fn make_nothing(&mut self){
        self.event_type = EventType::Nothing;
    }
    pub fn set_block(position:glm::Vec3,ray_cast_direction:glm::Vec3, block:BlockId)->Self{
        Self {
            event_type: EventType::SetBlock,
            vec3_slot0: position,
            u32_slot0: block.id(),
            vec3_slot1: ray_cast_direction,
            u32_slot1: 0,
            uvec3_slot1: Default::default()
        }
    }
    pub fn break_block(position:glm::Vec3,ray_cast_direction:glm::Vec3)->Self{
        Self::set_block(position,ray_cast_direction,AIR)
    }
    pub fn throw(position:glm::Vec3,velocity:glm::Vec3)->Self{
        Self{
            event_type: EventType::Throw,
            vec3_slot0: position,
            u32_slot0: 0,
            vec3_slot1: velocity,
            u32_slot1: 0,
            uvec3_slot1: Default::default()
        }
    }
}