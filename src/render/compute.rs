use crate::render::descriptors::{Descriptors, DescriptorsBuilder};
use crate::render::buffer::{Buffer, Gpu, Storage};
use ash::vk;
use std::marker::PhantomData;
use crate::render::device::Device;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::shader_module::ShaderModule;
use crate::render::shader_module::Compute as ShCompute;
use std::ffi::CString;
use failure::err_msg;
use ash::version::DeviceV1_0;
use ash::vk::{Pipeline, PipelineLayout};
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};

pub struct ComputePipelineBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    descriptors: Vec<vk::DescriptorBufferInfo>,
    shader: Option<(CString, ShaderModule<ShCompute>)>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct StorageBufferBinding<T: Copy>(u32, PhantomData<T>);

impl ComputePipelineBuilder {
    pub fn new() -> Self {
        Self { bindings: Vec::new(), descriptors: Vec::new(), shader: None }
    }

    pub fn shader(&mut self, name: &str, shader: ShaderModule<ShCompute>) -> &mut Self {
        self.shader.insert((CString::new(name).expect("Compute shader's function name contains null character"), shader));
        self
    }
    pub fn storage_buffer<T: Copy>(&mut self, buffer: &Buffer<T, Storage>) -> StorageBufferBinding<T> {
        let new_index = self.bindings.len() as u32;
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding: new_index,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(buffer.descriptor_info());
        StorageBufferBinding(new_index, PhantomData)
    }

    pub fn build(&self, device: &Device) -> Result<ComputePipeline, failure::Error> {
        let Self { bindings, descriptors, shader } = self;
        let (shader_name, shader_module) = shader.as_ref().ok_or_else(|| err_msg("No compute shader was specified"))?;
        let descriptor_layout = DescriptorLayout::new(device, bindings)?;
        let descriptor_layout_raw = descriptor_layout.raw();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(shader_name.as_c_str())
            .module(shader_module.raw());
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_layout_raw));
        let pipeline_layout = unsafe { device.inner().create_pipeline_layout(&pipeline_layout_create_info, None) }?;
        let p = vk::ComputePipelineCreateInfo::builder()
            .stage(stage.build())
            .layout(pipeline_layout);
        let result = unsafe {
            device.inner().create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        };

        match result {
            Ok(pipeline) => {
                ComputePipeline::new(ComputePipelineInner::new(pipeline, pipeline_layout, device), descriptor_layout, descriptors)
            }
            Err((pipeline, err)) => {
                ComputePipelineInner::new(pipeline, pipeline_layout, device);
                Err(err_msg(err))
            }
        }
    }
}

struct ComputePipelineInner {
    raw: Pipeline,
    layout: PipelineLayout,
    device: Device,
}

impl ComputePipelineInner {
    fn device(&self) -> &Device {
        &self.device
    }
    fn new(pipeline: Vec<vk::Pipeline>, pipeline_layout: PipelineLayout, device: &Device) -> Self {
        Self {
            raw: pipeline.into_iter().next().unwrap(),
            layout: pipeline_layout,
            device: device.clone(),
        }
    }
}

impl Drop for ComputePipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device().inner().destroy_pipeline(self.raw, None);
            self.device().inner().destroy_pipeline_layout(self.layout, None);
            // Safety: The pipeline is dropped first.
        }
    }
}

pub struct ComputePipeline {
    descriptor_set: DescriptorSet,
    descriptor_pool: DescriptorPool,
    descriptor_layout: DescriptorLayout,
    // Just keeping reference to prevent drop
    inner: ComputePipelineInner,
}

impl ComputePipeline {
    fn new(inner: ComputePipelineInner, descriptor_layout: DescriptorLayout, descriptors:&[vk::DescriptorBufferInfo]) -> Result<Self, failure::Error> {
        let descriptor_pool = DescriptorPool::manual_new(&descriptor_layout, 1, inner.device())?;
        let descriptor_sets = descriptor_pool.create_sets_with_same_layout(descriptor_layout.clone(), 1)?;
        let descriptor_set = descriptor_sets.into_iter().next().unwrap();
        for (binding, descriptor) in descriptors.iter().enumerate() {
            unsafe {
                descriptor_set.update_storage_buffer_raw(binding as u32, descriptor);
            }
        }
        Ok(Self { inner, descriptor_layout, descriptor_pool, descriptor_set })
    }
    pub fn device(&self) -> &Device {
        self.inner.device()
    }
    pub fn raw(&self) -> vk::Pipeline {
        self.inner.raw
    }
    pub fn layout(&self) -> vk::PipelineLayout {
        self.inner.layout
    }
    pub fn descriptor_layout(&self) -> &DescriptorLayout {
        &self.descriptor_layout
    }
    pub fn descriptor_set(&self) -> &DescriptorSet {
        &self.descriptor_set
    }
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        &self.descriptor_pool
    }
}
