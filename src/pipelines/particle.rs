use crate::render::data::VertexSource;
use ash::vk::VertexInputAttributeDescription;
use ash::vk;
use rand::random;
use crate::render::data::VertexAttrib;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Particle{
    pub position:glm::Vec3,
    pub sender_entity_idx:u32,
    pub velocity:glm::Vec3,
    pub energy:f32,
}
impl Particle{
    fn rand_f32()->f32{
        random::<f32>()*2.-1.
    }
    pub fn rand_vec3()->glm::Vec3{
        glm::vec3(Self::rand_f32(),Self::rand_f32(),Self::rand_f32())
    }
    pub fn random()->Self{
        let position = glm::vec3(random::<f32>()*16.,3.+random::<f32>()*8.,random::<f32>()*16.);
        Self{
            position,
            sender_entity_idx: 0,
            velocity: Self::rand_vec3()*0.1,
            energy: random::<f32>(),
        }
    }
    pub fn new(pos:glm::Vec3)->Self{
        Self{
            position: pos,
            sender_entity_idx: 0,
            velocity: glm::zero(),
            energy: random::<f32>()
        }
    }
}
impl VertexSource for Particle{
    fn get_attribute_descriptions(binding: u32) -> Vec<VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding,
                location: 0,
                format:  glm::Vec3::FORMAT,
                offset: offset_of!(Self, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  f32::FORMAT,
                offset: offset_of!(Self, energy) as u32,
            }
        ]
    }
}