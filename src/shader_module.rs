use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use ash::prelude::VkResult;
use std::ffi::{CStr, CString};
use std::rc::Rc;

pub struct ShaderModuleInner{
    m:vk::ShaderModule,
    stage:vk::ShaderStageFlags,
    device:Device
}
#[derive(Clone)]
pub struct ShaderModule{
   inner:Rc<Box<ShaderModuleInner>>
}
impl ShaderModule {
    pub fn new(src: &[u32], stage:vk::ShaderStageFlags, device: &Device) -> VkResult<Self> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(src);
        unsafe { device.device().create_shader_module(&shader_module_create_info, None) }
            .map(|m| Self { inner:Rc::new(Box::new(ShaderModuleInner{m,device:device.clone(),stage }))})
    }

    pub fn to_stage_info<'a>(&'a self, main_function:&'a CStr)->vk::PipelineShaderStageCreateInfoBuilder<'a>{
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.inner.m)
            .name(main_function)
            .stage(self.inner.stage)

    }
}

impl Drop for ShaderModuleInner{
    fn drop(&mut self) {
        unsafe { self.device.device().destroy_shader_module(self.m, None); }
    }
}