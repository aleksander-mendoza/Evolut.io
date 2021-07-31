use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use ash::prelude::VkResult;
use std::ffi::{CStr, CString};
use std::rc::Rc;
use ash::vk::ShaderStageFlags;
use std::marker::PhantomData;

pub trait ShaderStage{
    const STAGE:vk::ShaderStageFlags;
}

pub struct Vertex{}
impl ShaderStage for Vertex{
    const STAGE: ShaderStageFlags = ShaderStageFlags::VERTEX;
}

pub struct Fragment{}
impl ShaderStage for Fragment{
    const STAGE: ShaderStageFlags = ShaderStageFlags::FRAGMENT;
}

pub struct Compute{}
impl ShaderStage for Compute{
    const STAGE: ShaderStageFlags = ShaderStageFlags::COMPUTE;
}

pub struct ShaderModuleInner{
    m:vk::ShaderModule,
    device:Device,

}
#[derive(Clone)]
pub struct AnyShaderModule{
    inner:Rc<ShaderModuleInner>,
    stage: ShaderStageFlags
}
impl AnyShaderModule{
    pub fn to_stage_info<'a>(&'a self, main_function:&'a CStr)->vk::PipelineShaderStageCreateInfoBuilder<'a>{
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.inner.m)
            .name(main_function)
            .stage(self.stage)
    }
}

#[derive(Clone)]
pub struct ShaderModule<T:ShaderStage>{
   inner:Rc<ShaderModuleInner>,
    _p:PhantomData<T>
}


impl <T:ShaderStage> ShaderModule<T> {
    pub fn new(src: &[u32], device: &Device) -> VkResult<Self> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(src.into());
        unsafe { device.inner().create_shader_module(&shader_module_create_info, None) }
            .map(|m| Self { inner:Rc::new(ShaderModuleInner{m,device:device.clone() }),_p:PhantomData})
    }
    pub fn raw(&self)->vk::ShaderModule{
        self.inner.m
    }
    pub fn to_stage_info<'a>(&'a self, main_function:&'a CStr)->vk::PipelineShaderStageCreateInfoBuilder<'a>{
        vk::PipelineShaderStageCreateInfo::builder()
            .module(self.inner.m)
            .name(main_function)
            .stage(T::STAGE)

    }
    pub unsafe fn as_any(&self)->AnyShaderModule{
        AnyShaderModule{inner:self.inner.clone(), stage: T::STAGE }
    }
    pub unsafe fn into_any(self)->AnyShaderModule{
        let Self{inner,..} = self;
        AnyShaderModule{inner, stage: T::STAGE }
    }
}

impl Drop for ShaderModuleInner{
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_shader_module(self.m, None); }
    }
}