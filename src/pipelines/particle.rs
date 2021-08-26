use crate::render::data::VertexSource;
use ash::vk::VertexInputAttributeDescription;
use ash::vk;
use rand::random;
use crate::render::data::VertexAttrib;

#[repr(C, packed)]
#[derive(Copy,Clone,Debug)]
pub struct Particle{
    pub old_position:glm::Vec3,
    pub size:f32,
    pub new_position:glm::Vec3,
    pub color:f32,
}
impl Particle{
    fn rand_f32()->f32{
        random::<f32>()*2.-1.
    }
    pub fn rand_vec3()->glm::Vec3{
        glm::vec3(Self::rand_f32(),Self::rand_f32(),Self::rand_f32())
    }
    pub fn random()->Self{
        let new_position = glm::vec3(random::<f32>()*16.,3.+random::<f32>()*8.,random::<f32>()*16.);
        Self{
            new_position,
            size: 50.+random::<f32>()*50.,
            old_position: new_position + Self::rand_vec3()*0.1,
            color: random::<f32>(),
        }
    }
    pub fn new(pos:glm::Vec3)->Self{
        Self{
            old_position: pos,
            size: 50.+random::<f32>()*50.,
            new_position: pos,
            color: random::<f32>()
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
                offset: offset_of!(Self, new_position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 1,
                format:  f32::FORMAT,
                offset: offset_of!(Self, size) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding,
                location: 2,
                format:  f32::FORMAT,
                offset: offset_of!(Self, color) as u32,
            }
        ]
    }
}