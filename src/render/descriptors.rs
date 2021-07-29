use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::sampler::Sampler;
use crate::render::buffer::{Buffer, Uniform};
use crate::render::imageview::{ImageView, Color};
use ash::vk::{DescriptorBufferInfo, DescriptorSetLayoutBinding, DescriptorImageInfo};
use crate::render::descriptor_binding::DescriptorBinding;
use crate::render::device::Device;
use crate::render::swap_chain::{SwapChain, SwapchainImageIdx};
use crate::render::frames_in_flight::FramesInFlight;
use crate::render::host_buffer::HostBuffer;
use ash::vk;
use std::marker::PhantomData;


enum DescriptorUniform {
    Sampler(DescriptorImageInfo),
    Buffer(/*type size*/usize, /*element count*/usize),
}

pub struct DescriptorsBuilder {
    bindings: Vec<DescriptorSetLayoutBinding>,
    descriptors: Vec<DescriptorUniform>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct UniformBufferBinding<T, const size: usize>{
    binding:usize,
    _p:PhantomData<[T;size]>
}
impl <T, const size: usize> UniformBufferBinding<T,size>{
    fn new(binding:usize)->Self{
        Self{
            binding,
            _p:PhantomData
        }
    }
}

impl DescriptorsBuilder {
    pub fn new() -> Self {
        Self {
            bindings: vec![],
            descriptors: vec![],
        }
    }

    pub fn sampler(&mut self, sampler: &Sampler, image_view: &ImageView<Color>) {
        self.bindings.push(sampler.create_binding(self.descriptors.len() as u32));
        self.descriptors.push(DescriptorUniform::Sampler(sampler.descriptor_info(image_view)));
    }

    pub fn singleton_uniform_buffer<T: Copy>(&mut self, buffer: &T) -> UniformBufferBinding<T, 1> {
        self.array_uniform_buffer(std::array::from_ref(buffer))
    }
    pub fn array_uniform_buffer<T: Copy, const size: usize>(&mut self, buffer: &[T; size]) -> UniformBufferBinding<T, size> {
        let new_idx = self.descriptors.len();
        self.bindings.push(buffer.create_binding(new_idx as u32));
        self.descriptors.push(DescriptorUniform::Buffer(std::mem::size_of::<T>(), size));
        UniformBufferBinding::new(new_idx)
    }

    pub fn make_layout(self, device: &Device) -> Result<DescriptorsBuilderLocked, vk::Result> {
        let Self { bindings, descriptors } = self;
        DescriptorLayout::new(device, &bindings).map(move |descriptor_layout| DescriptorsBuilderLocked { descriptors, descriptor_layout })
    }
}

pub struct DescriptorsBuilderLocked {
    descriptors: Vec<DescriptorUniform>,
    descriptor_layout: DescriptorLayout,
}

impl DescriptorsBuilderLocked {
    pub fn layout(&self) -> &DescriptorLayout {
        &self.descriptor_layout
    }
    pub fn build(&self, swapchain: &SwapChain) -> Result<Descriptors, failure::Error> {
        let descriptor_pool = DescriptorPool::new(self.layout(), swapchain)?;
        let descriptor_sets = descriptor_pool.create_sets_with_same_layout(self.layout().clone(), swapchain.len())?;
        let uniform_buffers = self.uniform_buffers_per_binding(swapchain)?;
        for (frame, descriptor_set) in descriptor_sets.iter().enumerate() {
            for (binding, descriptor) in self.descriptors.iter().enumerate() {
                unsafe {
                    match descriptor {
                        DescriptorUniform::Sampler(sampler_info) => {
                            descriptor_set.update_sampler_raw(binding as u32, sampler_info);
                        }
                        &DescriptorUniform::Buffer(type_size, elem_count) => {
                            descriptor_set.update_buffer_raw(binding as u32, &uniform_buffers[binding].buffer_per_frame[frame].buffer().descriptor_info());
                        }
                    }
                }
            }
        }

        Ok(Descriptors { descriptor_layout: self.layout().clone(), uniform_buffers, descriptor_pool, descriptor_sets })
    }
    fn uniform_buffers_per_binding(&self, swapchain: &SwapChain) -> Result<Vec<UniformBuffers>,vk::Result> {
        self.descriptors.iter().map(|descriptor| UniformBuffers::new(descriptor, swapchain)).collect()
    }
}

pub struct UniformBuffers {
    buffer_per_frame: Vec<HostBuffer<u8, Uniform>>,
}

impl UniformBuffers {
    fn new(descriptor:&DescriptorUniform, swapchain: &SwapChain) -> Result<UniformBuffers, vk::Result> {
        match descriptor {
            DescriptorUniform::Sampler(_) => {
                Ok(Self::sampler())
            }
            &DescriptorUniform::Buffer(type_size, elem_count) => {
                Self::buffer(swapchain, type_size, elem_count)
            }
        }
    }
    fn sampler() -> Self {
        Self { buffer_per_frame: Vec::new() }
    }
    fn buffer(swapchain: &SwapChain, type_size: usize, elem_count: usize) -> Result<Self,vk::Result> {
        let buffers: Result<Vec<HostBuffer<u8, Uniform>>, vk::Result> = (0..swapchain.len()).into_iter()
            .map(|_| HostBuffer::with_capacity(swapchain.device(), type_size * elem_count)).collect();
        Ok(Self { buffer_per_frame: buffers? })
    }
}

pub struct Descriptors {
    descriptor_layout: DescriptorLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<DescriptorSet>,
    uniform_buffers: Vec<UniformBuffers>,
}

impl Descriptors {
    pub fn descriptor_layout(&self) -> &DescriptorLayout {
        &self.descriptor_layout
    }
    pub fn descriptor_pool(&self) -> &DescriptorPool {
        &self.descriptor_pool
    }
    pub fn descriptor_set(&self, image_idx:SwapchainImageIdx) -> &DescriptorSet {
        &self.descriptor_sets[image_idx.get_usize()]
    }
    pub fn uniform_as_slice_mut<T,const size:usize>(&mut self, image_idx:SwapchainImageIdx, buffer:UniformBufferBinding<T,size>) -> &mut [T;size]{
        let bytes:&mut[u8] = self.uniform_buffers[buffer.binding].buffer_per_frame[image_idx.get_usize()].as_slice_mut();
        debug_assert_eq!(bytes.len(),std::mem::size_of::<T>()*size);
        let bytes = bytes.as_mut_ptr() as *mut [T;size];
        unsafe{&mut *bytes}
    }
    pub fn uniform_as_slice<T,const size:usize>(&self, image_idx:SwapchainImageIdx, buffer:UniformBufferBinding<T,size>) -> &[T;size]{
        let bytes:&[u8] = self.uniform_buffers[buffer.binding].buffer_per_frame[image_idx.get_usize()].as_slice();
        debug_assert_eq!(bytes.len(),std::mem::size_of::<T>()*size);
        let bytes = bytes.as_ptr() as *const [T;size];
        unsafe{&*bytes}
    }
}