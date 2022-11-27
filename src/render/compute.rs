

use ash::vk;
use std::marker::PhantomData;
use crate::render::device::Device;
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::shader_module::ShaderModule;
use crate::render::shader_module::Compute as ShCompute;
use std::ffi::CString;
use failure::err_msg;
use ash::vk::{Pipeline, PipelineLayout};
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::buffer_type::{AsStorage, Uniform};
use crate::render::buffer::{descriptor_info, Buffer};
use std::rc::Rc;

pub struct ComputeDescriptorsBuilder{
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    descriptors: Vec<vk::DescriptorBufferInfo>,
}

impl ComputeDescriptorsBuilder{
    pub fn new() -> Self {
        Self { bindings: Vec::new(), descriptors: Vec::new() }
    }
    pub fn storage_buffer<V: Copy, T:AsStorage>(&mut self, buffer: &impl Buffer<V, T>) -> StorageBufferBinding<V> {
        let new_index = self.bindings.len() as u32;
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding: new_index,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(descriptor_info(buffer));
        StorageBufferBinding(new_index, PhantomData)
    }

    pub fn uniform_buffer<V: Copy>(&mut self, buffer: &impl Buffer<V,Uniform> ) -> UniformBufferBinding<V> {
        let new_index = self.bindings.len() as u32;
        self.bindings.push(vk::DescriptorSetLayoutBinding {
            binding: new_index,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: buffer.len() as u32,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        });
        self.descriptors.push(descriptor_info(buffer));
        UniformBufferBinding(new_index, PhantomData)
    }

    pub fn build(&self, device: &Device) -> Result<ComputeDescriptors, failure::Error> {
        let Self { bindings, descriptors } = self;
        let descriptor_layout = DescriptorLayout::new(device, bindings)?;
        let descriptor_pool = DescriptorPool::manual_new(&descriptor_layout, 1, device)?;
        let descriptor_sets = descriptor_pool.create_sets_with_same_layout(descriptor_layout.clone(), 1)?;
        let descriptor_set = descriptor_sets.into_iter().next().unwrap();
        for (descriptor, binding) in descriptors.iter().zip(bindings.iter()) {
            unsafe {
                if binding.descriptor_type == vk::DescriptorType::STORAGE_BUFFER{
                    descriptor_set.update_storage_buffer_raw(binding.binding, descriptor);
                }else{
                    debug_assert_eq!(binding.descriptor_type , vk::DescriptorType::UNIFORM_BUFFER);
                    descriptor_set.update_uniform_buffer_raw(binding.binding, descriptor);
                }

            }
        }
        Ok(ComputeDescriptors{inner:Rc::new(ComputeDescriptorsInner{
            descriptor_set,
            descriptor_pool,
            descriptor_layout,
        })})
    }
}

struct ComputeDescriptorsInner{
    descriptor_set: DescriptorSet,
    descriptor_pool: DescriptorPool,
    descriptor_layout: DescriptorLayout,
}
#[derive(Clone)]
pub struct ComputeDescriptors{
    inner:Rc<ComputeDescriptorsInner>
}

impl ComputeDescriptors{
    pub fn device(&self) -> &Device {
        self.inner.descriptor_pool.device()
    }
    pub fn descriptor_layout(&self) -> &DescriptorLayout {
        &self.inner.descriptor_layout
    }
    pub fn descriptor_set(&self) -> &DescriptorSet {
        &self.inner.descriptor_set
    }
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        &self.inner.descriptor_pool
    }
    pub fn build(&self, shader_name: &str, shader_module: ShaderModule<ShCompute>, specialization:&vk::SpecializationInfoBuilder) -> Result<ComputePipeline, failure::Error> {
        let shader_name = CString::new(shader_name).expect("Compute shader's function name contains null character");
        let descriptor_layout_raw = self.descriptor_layout().raw();
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(shader_name.as_c_str())
            .specialization_info(specialization)
            .module(shader_module.raw());
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_layout_raw));
        let pipeline_layout = unsafe { self.device().inner().create_pipeline_layout(&pipeline_layout_create_info, None) }?;
        let p = vk::ComputePipelineCreateInfo::builder()
            .stage(stage.build())
            .layout(pipeline_layout);
        let result = unsafe {
            self.device().inner().create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        };

        match result {
            Ok(pipeline) => {
                ComputePipeline::new(ComputePipelineInner::new(pipeline, pipeline_layout, self.device()), self.clone())
            }
            Err((pipeline, err)) => {
                ComputePipelineInner::new(pipeline, pipeline_layout, self.device());
                Err(err_msg(err))
            }
        }
    }
}


#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct StorageBufferBinding<T: Copy>(u32, PhantomData<T>);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct UniformBufferBinding<T: Copy>(u32, PhantomData<T>);

struct ComputePipelineInner {
    raw: Pipeline,
    layout: PipelineLayout,
    device:Device
}

impl ComputePipelineInner {
    fn new(pipeline: Vec<vk::Pipeline>, pipeline_layout: PipelineLayout, device:&Device) -> Self {
        Self {
            raw: pipeline.into_iter().next().unwrap(),
            layout: pipeline_layout,
            device:device.clone(),
        }
    }
    pub fn device(&self) -> &Device {
        &self.device
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
    descriptors:ComputeDescriptors,
    // Just keeping reference to prevent drop
    inner: ComputePipelineInner,
}

impl ComputePipeline {
    fn new(inner: ComputePipelineInner, descriptors:ComputeDescriptors) -> Result<Self, failure::Error> {
        Ok(Self { inner, descriptors })
    }
    pub fn device(&self) -> &Device {
        self.descriptors.device()
    }
    pub fn raw(&self) -> vk::Pipeline {
        self.inner.raw
    }
    pub fn layout(&self) -> vk::PipelineLayout {
        self.inner.layout
    }
    pub fn descriptor_layout(&self) -> &DescriptorLayout {
        self.descriptors.descriptor_layout()
    }
    pub fn descriptor_set(&self) -> &DescriptorSet {
        self.descriptors.descriptor_set()
    }
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        self.descriptors.descriptor_pool()
    }
}
