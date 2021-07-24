use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use std::marker::PhantomData;
use crate::render::render_pass::RenderPass;
use ash::vk::ClearValue;
use crate::render::pipeline::Pipeline;
use crate::render::semaphore::Semaphore;
use ash::prelude::VkResult;
use crate::render::fence::Fence;
use crate::render::buffer::{Buffer, Type, Gpu, Cpu, GpuIndirect};
use crate::render::data::VertexSource;
use crate::render::descriptor_pool::DescriptorSet;
use crate::render::texture::{Dim, Texture};
use crate::render::framebuffer::Framebuffer;
use crate::render::imageview::{Color, Aspect};

pub trait OptionalRenderPass {}

pub struct StateClear {}

pub struct StateBegan {}

pub struct StateRenderPassBegan {}

pub struct StateFinished {}

impl OptionalRenderPass for StateBegan {}

impl OptionalRenderPass for StateRenderPassBegan {}

pub struct CommandBuffer<State> {
    raw: vk::CommandBuffer,
    device: Device,
    _state: PhantomData<State>,
}

impl<X> CommandBuffer<X> {
    fn state_transition<Y>(self) -> CommandBuffer<Y> {
        let Self { raw, device, .. } = self;
        CommandBuffer { raw, device, _state: PhantomData }
    }
}


impl<X: OptionalRenderPass> CommandBuffer<X> {
    pub fn copy<V, T1: Type, T2: Type>(self, src: &Buffer<V, T1>, dst: &Buffer<V, T2>) -> Self {
        assert_eq!(src.capacity(), dst.capacity());
        unsafe {
            self.device.inner().cmd_copy_buffer(
                self.raw,
                src.raw(),
                dst.raw(),
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: src.mem_capacity(),
                }],
            );
        }
        self
    }

    pub fn copy_to_image<V, T: Type, D: Dim>(self, src: &Buffer<V, T>, dst: &Texture<D, Color>, img_layout: vk::ImageLayout) -> Self {
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
            );
        }
        self
    }

    pub fn layout_barrier<D: Dim, A: Aspect>(self, image: &Texture<D, A>, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> Self {
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
            );
        }
        self
    }
}

impl CommandBuffer<StateClear> {
    pub fn begin(self, usage: vk::CommandBufferUsageFlags) -> Result<CommandBuffer<StateBegan>, vk::Result> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(usage);
        let result = unsafe {
            self.device.inner().begin_command_buffer(self.raw, &command_buffer_begin_info)
        };
        result.map(move |()| self.state_transition())
    }

    pub fn single_pass(self,
                       usage: vk::CommandBufferUsageFlags,
                       render_pass: &RenderPass,
                       framebuffer: &Framebuffer,
                       render_area: vk::Rect2D,
                       clear: &[ClearValue],
                       pipeline: &Pipeline,
                       vertex_count: u32,
                       instance_count: u32,
                       first_vertex: u32,
                       first_instance: u32) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .draw(vertex_count, instance_count, first_vertex, first_instance)
            .end_render_pass()
            .end()
    }

    pub fn single_pass_vertex_input<V: VertexSource, T: Type>(self,
                                                              usage: vk::CommandBufferUsageFlags,
                                                              render_pass: &RenderPass,
                                                              framebuffer: &Framebuffer,
                                                              render_area: vk::Rect2D,
                                                              clear: &[ClearValue],
                                                              pipeline: &Pipeline,
                                                              buffer: &Buffer<V, T>) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .vertex_input(buffer)
            .draw(buffer.capacity() as u32, 1, 0, 0)
            .end_render_pass()
            .end()
    }

    pub fn single_pass_vertex_input_uniform<V: VertexSource, T: Type>(self,
                                                                      usage: vk::CommandBufferUsageFlags,
                                                                      render_pass: &RenderPass,
                                                                      framebuffer: &Framebuffer,
                                                                      uniform: &DescriptorSet,
                                                                      render_area: vk::Rect2D,
                                                                      clear: &[ClearValue],
                                                                      pipeline: &Pipeline,
                                                                      buffer: &Buffer<V, T>) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .vertex_input(buffer)
            .uniform(pipeline, uniform)
            .draw(buffer.capacity() as u32, 1, 0, 0)
            .end_render_pass()
            .end()
    }

    pub fn single_pass_indirect_input_uniform<V: VertexSource, T: Type>(self,
                                                                        usage: vk::CommandBufferUsageFlags,
                                                                        render_pass: &RenderPass,
                                                                        framebuffer: &Framebuffer,
                                                                        uniform: &DescriptorSet,
                                                                        render_area: vk::Rect2D,
                                                                        clear: &[ClearValue],
                                                                        pipeline: &Pipeline,
                                                                        buffer: &Buffer<V, T>,
                                                                        indirect_buffer:&Buffer<vk::DrawIndirectCommand,GpuIndirect>) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .vertex_input(buffer)
            .uniform(pipeline, uniform)
            .draw_indirect(indirect_buffer)
            .end_render_pass()
            .end()
    }
}

impl CommandBuffer<StateBegan> {
    pub fn render_pass(self, render_pass: &RenderPass, framebuffer: &Framebuffer, render_area: vk::Rect2D, clear: &[ClearValue]) -> CommandBuffer<StateRenderPassBegan> {
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
        self.state_transition()
    }
    pub fn end(self) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        let result = unsafe {
            self.device.inner().end_command_buffer(
                self.raw,
            )
        };
        result.map(move |()| self.state_transition())
    }
}

impl CommandBuffer<StateRenderPassBegan> {
    pub fn bind_pipeline(self, pipeline: &Pipeline) -> Self {
        unsafe {
            self.device.inner().cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            );
        }
        self
    }
    pub fn vertex_input<V: VertexSource, T: Type>(self, buffer: &Buffer<V, T>) -> Self {
        unsafe {
            self.device.inner().cmd_bind_vertex_buffers(
                self.raw,
                0,
                &[buffer.raw()],
                &[0],
            );
        }
        self
    }
    pub fn uniform(self, pipeline: &Pipeline, uniform: &DescriptorSet) -> Self {
        unsafe {
            self.device.inner().cmd_bind_descriptor_sets(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout(),
                0,
                &[uniform.raw()],
                &[],
            );
        }
        self
    }
    pub fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self {
        unsafe {
            self.device.inner().cmd_draw(
                self.raw,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
        self
    }
    pub fn draw_indirect(self, buffer: &Buffer<vk::DrawIndirectCommand, GpuIndirect>) -> Self {
        unsafe {
            self.device.inner().cmd_draw_indirect(
                self.raw,
                buffer.raw(),
                buffer.offset() as u64,
                buffer.len() as u32,
                std::mem::size_of::<vk::DrawIndirectCommand>() as u32,
            );
        }
        self
    }
    pub fn end_render_pass(self) -> CommandBuffer<StateBegan> {
        unsafe {
            self.device.inner().cmd_end_render_pass(
                self.raw,
            );
        }
        self.state_transition()
    }
}

impl CommandBuffer<StateFinished> {
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
                self.device.raw_queue(),
                std::slice::from_ref(&submit_infos),
                fence_to_signal.map(Fence::raw).unwrap_or(vk::Fence::null()),
            )
        }
    }
}

pub struct CommandPool {
    raw: vk::CommandPool,
    device: Device,
}

impl CommandPool {
    pub fn new(device: &Device) -> Result<Self, vk::Result> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(device.family_index());
        unsafe {
            device.inner().create_command_pool(&command_pool_create_info, None)
        }.map(|raw| Self { raw, device: device.clone() })
    }
    pub fn clear(&mut self) -> VkResult<()> {
        unsafe { self.device.inner().reset_command_pool(self.raw, vk::CommandPoolResetFlags::RELEASE_RESOURCES) }
    }
    pub fn create_command_buffer(&self) -> Result<CommandBuffer<StateClear>, vk::Result> {
        self.create_command_buffers(1).map(|v| v.into_iter().next().unwrap())
    }
    pub fn create_command_buffers(&self, count: u32) -> Result<Vec<CommandBuffer<StateClear>>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw)
            .command_buffer_count(count)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            self.device.inner()
                .allocate_command_buffers(&command_buffer_allocate_info)
        }.map(|vec| vec.into_iter().map(|raw| CommandBuffer { raw, device: self.device.clone(), _state: PhantomData }).collect())
    }
    pub fn free(&self, cmd: CommandBuffer<StateFinished>) {
        unsafe {
            self.device.inner().free_command_buffers(self.raw, &[cmd.raw])
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_command_pool(self.raw, None) }
    }
}