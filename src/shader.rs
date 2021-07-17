use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use ash::prelude::VkResult;

pub struct ShaderModule{
    m:vk::ShaderModule
}
impl ShaderModule {
    pub fn create_shader_module(src: &[u32], device: &Device) -> VkResult<Self> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(src);
        unsafe { device.device().create_shader_module(&shader_module_create_info, None) }.map(|m| Self { m })
    }
}