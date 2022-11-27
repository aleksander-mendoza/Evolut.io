use ash::vk;
use crate::render::device::Device;

use crate::render::render_pass::RenderPass;
use ash::vk::{ClearValue, CommandPoolResetFlags};
use crate::render::pipeline::{Pipeline, PushConstant, BufferBinding};
use crate::render::semaphore::Semaphore;
use ash::prelude::VkResult;
use crate::render::fence::Fence;
use crate::render::owned_buffer::{OwnedBuffer};
use crate::render::data::{VertexSource, VertexAttrib};
use crate::render::descriptor_pool::DescriptorSet;
use crate::render::texture::{Dim, Texture};
use crate::render::framebuffer::Framebuffer;
use crate::render::imageview::{Color, Aspect};
use std::rc::Rc;
use crate::render::util::any_as_u8_slice;
use crate::render::stage_buffer::StageBuffer;
use crate::render::compute::{ComputePipeline};
use crate::render::buffer_type::{GpuWriteable, CpuWriteable, GpuIndirect, BufferType, AsStorage};
use crate::render::buffer::{Buffer, make_buffer_barrier};

pub struct CommandBuffer {
    raw: vk::CommandBuffer,
    device: Device,
    queue_idx:usize,
}

impl CommandBuffer {
    pub fn raw(&self) -> vk::CommandBuffer{
        self.raw
    }
    pub fn copy_from_staged_if_has_changes<V:Copy, C:CpuWriteable, G:GpuWriteable, B:Buffer<V,G>>(&mut self, staged: &mut StageBuffer<V, C, G, B>) -> &mut Self {
        if staged.has_unflushed_changes(){
            staged.mark_with_no_changes();
            self.copy_from_staged(staged)
        }else{
            self
        }
    }
    pub fn copy_from_staged<V:Copy, C:CpuWriteable, G:GpuWriteable, B:Buffer<V,G>>(&mut self, staged: &StageBuffer<V, C, G, B>) -> &mut Self {
        self.copy(staged.cpu(), staged.gpu())

    }

    pub fn copy_to_staged<V:Copy, C:CpuWriteable, G:GpuWriteable, B:Buffer<V,G>>(&mut self, staged: &StageBuffer<V, C, G, B>) -> &mut Self {
        self.copy(staged.gpu(), staged.cpu())

    }


    pub fn copy<V:Copy, T1: BufferType, T2: BufferType>(&mut self, src: &impl Buffer<V, T1>, dst: &impl Buffer<V, T2>) -> &mut Self {
        assert!(src.bytes() <= dst.bytes());
        unsafe {
            self.device.inner().cmd_copy_buffer(
                self.raw,
                src.raw(),
                dst.raw(),
                &[vk::BufferCopy {
                    src_offset: src.offset(),
                    dst_offset: dst.offset(),
                    size: src.bytes(),
                }],
            )
        }
        self
    }

    pub fn copy_to_image<V:Copy, T: BufferType, D: Dim>(&mut self, src: &OwnedBuffer<V, T>, dst: &Texture<D, Color>, img_layout: vk::ImageLayout) -> &mut Self {
        // assert_eq!(src.capacity(),dst.capacity());
        unsafe {
            self.device.inner().cmd_copy_buffer_to_image(
                self.raw,
                src.raw(),
                dst.raw(),
                img_layout,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: dst.extent(),
                }],
            )
        }
        self
    }
    pub fn buffer_barrier<V:Copy,T:AsStorage>(&mut self, buffer:&OwnedBuffer<V,T>, src_access_mask: vk::AccessFlags, dst_access_mask: vk::AccessFlags, source_stage: vk::PipelineStageFlags, destination_stage: vk::PipelineStageFlags) -> &mut Self{
        unsafe {
            self.device.inner().cmd_pipeline_barrier(
                self.raw,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[make_buffer_barrier(buffer,src_access_mask, dst_access_mask)],
                &[],
            )
        }
        self
    }
    pub fn buffer_barriers(&mut self, source_stage: vk::PipelineStageFlags, destination_stage: vk::PipelineStageFlags, barriers: &[vk::BufferMemoryBarrier]) -> &mut Self{
        unsafe {
            self.device.inner().cmd_pipeline_barrier(
                self.raw,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                barriers,
                &[],
            )
        }
        self
    }
    pub fn execution_barrier(&mut self, source_stage: vk::PipelineStageFlags, destination_stage: vk::PipelineStageFlags) -> &mut Self{
        unsafe {
            self.device.inner().cmd_pipeline_barrier(
                self.raw,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            )
        }
        self
    }
    pub fn layout_barrier<D: Dim, A: Aspect>(&mut self, image: &Texture<D, A>, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> &mut Self{
        let src_access_mask;
        let dst_access_mask;
        let source_stage;
        let destination_stage;
        if old_layout == vk::ImageLayout::UNDEFINED && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("Unsupported layout transition!")
        }
        let image_barriers = vk::ImageMemoryBarrier::builder()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.raw())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.device.inner().cmd_pipeline_barrier(
                self.raw,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&image_barriers),
            )
        }
        self
    }

    pub fn begin(&mut self, usage: vk::CommandBufferUsageFlags) -> Result<&mut Self, vk::Result> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(usage);
        let result = unsafe {
            self.device.inner().begin_command_buffer(self.raw, &command_buffer_begin_info)
        };
        result.map(|()|self)
    }

    // pub fn single_pass(&mut self,
    //                    usage: vk::CommandBufferUsageFlags,
    //                    render_pass: &RenderPass,
    //                    framebuffer: &Framebuffer,
    //                    render_area: vk::Rect2D,
    //                    clear: &[ClearValue],
    //                    pipeline: &Pipeline,
    //                    vertex_count: u32,
    //                    instance_count: u32,
    //                    first_vertex: u32,
    //                    first_instance: u32) -> Result<&mut Self, vk::Result> {
    //     self.begin(usage)?
    //         .render_pass(render_pass, framebuffer, render_area, clear)
    //         .bind_pipeline(pipeline)
    //         .draw(vertex_count, instance_count, first_vertex, first_instance)
    //         .end_render_pass()
    //         .end()
    // }
    //
    // pub fn single_pass_vertex_input<V: VertexSource, T: BufferType>(&mut self,
    //                                                           usage: vk::CommandBufferUsageFlags,
    //                                                           render_pass: &RenderPass,
    //                                                           framebuffer: &Framebuffer,
    //                                                           render_area: vk::Rect2D,
    //                                                           clear: &[ClearValue],
    //                                                           pipeline: &Pipeline,
    //                                                           buffer: &Buffer<V, T>) -> Result<&mut Self, vk::Result> {
    //     self.begin(usage)?
    //         .render_pass(render_pass, framebuffer, render_area, clear)
    //         .bind_pipeline(pipeline)
    //         .vertex_input(buffer)
    //         .draw(buffer.capacity() as u32, 1, 0, 0)
    //         .end_render_pass()
    //         .end()
    // }
    //
    // pub fn single_pass_vertex_input_uniform<V: VertexSource, T: BufferType>(&mut self,
    //                                                                   usage: vk::CommandBufferUsageFlags,
    //                                                                   render_pass: &RenderPass,
    //                                                                   framebuffer: &Framebuffer,
    //                                                                   uniform: &DescriptorSet,
    //                                                                   render_area: vk::Rect2D,
    //                                                                   clear: &[ClearValue],
    //                                                                   pipeline: &Pipeline,
    //                                                                   buffer: &Buffer<V, T>) -> Result<&mut Self, vk::Result> {
    //     self.begin(usage)?
    //         .render_pass(render_pass, framebuffer, render_area, clear)
    //         .bind_pipeline(pipeline)
    //         .vertex_input(buffer)
    //         .uniform(pipeline, uniform)
    //         .draw(buffer.capacity() as u32, 1, 0, 0)
    //         .end_render_pass()
    //         .end()
    // }
    //

    pub fn render_pass(&mut self, render_pass: &RenderPass, framebuffer: &Framebuffer, render_area: vk::Rect2D, clear: &[ClearValue]) -> &mut Self{
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw())
            .framebuffer(framebuffer.raw())
            .render_area(render_area)
            .clear_values(clear);

        unsafe {
            self.device.inner().cmd_begin_render_pass(
                self.raw,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            )
        }
        self
    }
    pub fn end(&mut self) -> Result<&mut Self, vk::Result> {
        let result = unsafe {
            self.device.inner().end_command_buffer(
                self.raw,
            )
        };
        result.map( |()| self)
    }
    pub fn bind_pipeline(&mut self, pipeline: &Pipeline) -> &mut Self {
        unsafe {
            self.device.inner().cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            )
        }
        self
    }
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) -> &mut Self {
        unsafe {
            self.device.inner().cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.raw(),
            );
        }
        self
    }
    pub fn bind_compute_descriptors(&mut self, pipeline: &ComputePipeline) -> &mut Self {
        unsafe{
            self.device.inner().cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout(),
                0,
                &[pipeline.descriptor_set().raw()],
                &[],
            );
        }
        self
    }
    pub fn dispatch_1d(&mut self, x:u32) -> &mut Self {
        self.dispatch_2d(x,1)
    }
    pub fn dispatch_2d(&mut self, x:u32,y:u32) -> &mut Self {
        self.dispatch_3d(x,y,1)
    }
    pub fn dispatch_3d(&mut self, x:u32,y:u32,z:u32) -> &mut Self {
        unsafe {
            self.device.inner().cmd_dispatch(
                self.raw, x,y, z
            );
        }
        self
    }
    pub fn dispatch_indirect(&mut self, indirect_buffer:&impl Buffer<vk::DispatchIndirectCommand,GpuIndirect>, offset:vk::DeviceSize) -> &mut Self {
        unsafe {
            self.device.inner().cmd_dispatch_indirect(
                self.raw, indirect_buffer.raw(), indirect_buffer.element_offset(offset) as u64
            );
        }
        self
    }
    pub fn fill<T:BufferType>(&mut self, buffer:&impl Buffer<u32,T>, value:u32) -> &mut Self {
        unsafe {
            self.device.inner().cmd_fill_buffer(
                self.raw, buffer.raw(), buffer.offset(), buffer.bytes(), value
            );
        }
        self
    }
    pub fn fill_zeros<V:Copy, T:BufferType>(&mut self, buffer:&impl Buffer<V,T>) -> &mut Self {
        unsafe {
            self.device.inner().cmd_fill_buffer(
                self.raw, buffer.raw(), buffer.offset(), buffer.bytes(), 0
            );
        }
        self
    }
    pub fn push_constant<V: VertexAttrib>(&mut self,pipeline: &Pipeline, push_constant:PushConstant<V>, c: &V) -> &mut Self{

        unsafe {
            self.device.inner().cmd_push_constants(
                self.raw,
                pipeline.layout(),
                vk::ShaderStageFlags::VERTEX,
                push_constant.offset(),
                any_as_u8_slice(c)
            )
        }
        self
    }
    pub fn vertex_input<V: VertexSource, T: BufferType>(&mut self, binding:BufferBinding<V>, buffer: &impl Buffer<V, T>) -> &mut Self{
        unsafe {
            self.device.inner().cmd_bind_vertex_buffers(
                self.raw,
                binding.binding(),
                &[buffer.raw()],
                &[buffer.offset()],
            )
        }
        self
    }
    pub fn uniform(&mut self, pipeline: &Pipeline, uniform: &DescriptorSet) -> &mut Self{
        unsafe {
            self.device.inner().cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                0,
                &[uniform.raw()],
                &[],
            )
        }
        self
    }
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> &mut Self{
        unsafe {
            self.device.inner().cmd_draw(
                self.raw,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        }
        self
    }
    pub fn draw_indirect(&mut self, buffer: &impl Buffer<vk::DrawIndirectCommand, GpuIndirect>) ->&mut Self {
        unsafe {
            self.device.inner().cmd_draw_indirect(
                self.raw,
                buffer.raw(),
                buffer.offset(),
                buffer.len() as u32,
                std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
            )
        }
        self
    }
    pub fn end_render_pass(&mut self) -> &mut Self{
        unsafe {
            self.device.inner().cmd_end_render_pass(
                self.raw,
            )
        }
        self
    }
    pub fn submit(&self, wait_for: &[(&Semaphore, vk::PipelineStageFlags)], then_signal: &[Semaphore], fence_to_signal: Option<&Fence>) -> VkResult<()> {
        let wait_semaphores: Vec<vk::Semaphore> = wait_for.iter().map(|(s, _)| s.raw()).collect();
        let wait_stages: Vec<vk::PipelineStageFlags> = wait_for.iter().map(|(_, s)| *s).collect();
        let signal_semaphores: Vec<vk::Semaphore> = then_signal.iter().map(Semaphore::raw).collect();
        let submit_infos = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores.as_slice())
            .signal_semaphores(signal_semaphores.as_slice())
            .command_buffers(std::slice::from_ref(&self.raw))
            .wait_dst_stage_mask(wait_stages.as_slice());
        unsafe {
            self.device.inner().queue_submit(
                self.device.raw_queue(self.queue_idx),
                std::slice::from_ref(&submit_infos),
                fence_to_signal.map(Fence::raw).unwrap_or(vk::Fence::null()),
            )
        }
    }
    pub fn reset(&mut self) -> VkResult<&mut Self> {
        unsafe {
            self.device.inner().reset_command_buffer(self.raw, vk::CommandBufferResetFlags::empty())
        }.map(|()|self)
    }
}


struct CommandPoolInner {
    raw: vk::CommandPool,
    device: Device,
    queue_idx:usize,
}

impl Drop for CommandPoolInner {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_command_pool(self.raw, None) }
    }
}

#[derive(Clone)]
pub struct CommandPool {
    inner:Rc<CommandPoolInner>
}
// impl !Send for CommandPool{} //Command pool cannot be shared between threads!

impl CommandPool {
    pub fn queue_idx(&self) -> usize{self.inner.queue_idx}
    pub fn device(&self) -> &Device {
        &self.inner.device
    }
    pub fn raw(&self) -> vk::CommandPool {
        self.inner.raw
    }
    pub fn release(&self) -> VkResult<()> {
        unsafe{self.device().inner().reset_command_pool(self.raw(),CommandPoolResetFlags::RELEASE_RESOURCES)}
    }
    pub fn reset(&self) -> VkResult<()> {
        unsafe{self.device().inner().reset_command_pool(self.raw(),CommandPoolResetFlags::empty())}
    }
    pub fn new(device: &Device, queue_idx:usize, allow_resets:bool) -> Result<Self, vk::Result> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(device.family_index());
        let command_pool_create_info = if allow_resets{
            command_pool_create_info.flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        }else{
            command_pool_create_info
        };
        unsafe {
            device.inner().create_command_pool(&command_pool_create_info, None)
        }.map(|raw| Self { inner:Rc::new(CommandPoolInner{ queue_idx,raw, device: device.clone() })})
    }
    pub fn clear(&mut self) -> VkResult<()> {
        unsafe { self.device().inner().reset_command_pool(self.raw(), vk::CommandPoolResetFlags::RELEASE_RESOURCES) }
    }
    pub fn create_command_buffer(&self) -> Result<CommandBuffer, vk::Result> {
        self.create_command_buffers(1).map(|v| v.into_iter().next().unwrap())
    }
    pub fn create_command_buffers(&self, count: u32) -> Result<Vec<CommandBuffer>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw())
            .command_buffer_count(count)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            self.device().inner()
                .allocate_command_buffers(&command_buffer_allocate_info)
        }.map(|vec| vec.into_iter().map(|raw| CommandBuffer { queue_idx:self.queue_idx(), raw, device: self.device().clone() }).collect())
    }
    pub fn free(&self, cmd: CommandBuffer) {
        unsafe {
            self.device().inner().free_command_buffers(self.raw(), &[cmd.raw])
        }
    }
}
