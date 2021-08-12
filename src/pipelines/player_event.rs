
use crate::blocks::{Block, WorldSize};
use std::fmt::{Debug, Formatter};

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
                write!(f,"Throw{{position={}, velocity={}}}",self.vec3_slot0, self.vec3_slot1)
            }
            EventType::SetBlock => {
                write!(f,"SetBlock{{block_idx={}, block_id={}}}",self.u32_slot0, self.u32_slot1)
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
    pub fn set_block(block_idx:u32, block:&Block)->Self{
        Self {
            event_type: EventType::SetBlock,
            vec3_slot0: Default::default(),
            u32_slot0: block_idx,
            vec3_slot1: Default::default(),
            u32_slot1: block.id(),
            uvec3_slot1: Default::default()
        }
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