use ash::vk;
use crate::device::Device;
use ash::version::DeviceV1_0;
use std::marker::PhantomData;
use crate::render_pass::RenderPass;
use crate::imageview::Framebuffer;
use ash::vk::ClearValue;
use crate::pipeline::Pipeline;
use crate::semaphore::Semaphore;
use ash::prelude::VkResult;
use crate::fence::Fence;
use crate::buffer::{Buffer, Usage, SharingMode};
use crate::data::VertexSource;


pub struct StateClear {}

pub struct StateBegan {}

pub struct StateRenderPassBegan {}

pub struct StateFinished {}

pub struct CommandBuffer<State> {
    raw: vk::CommandBuffer,
    device: Device,
    _state: PhantomData<State>,
}

impl<X> CommandBuffer<X> {

    fn state_transition<Y>(self) -> CommandBuffer<Y> {
        let Self{raw, device, ..} = self;
        CommandBuffer{raw, device, _state:PhantomData}
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
            .finish()
    }

    pub fn single_pass_vertex_input<V:VertexSource, U:Usage, S:SharingMode>(self,
                       usage: vk::CommandBufferUsageFlags,
                       render_pass: &RenderPass,
                       framebuffer: &Framebuffer,
                       render_area: vk::Rect2D,
                       clear: &[ClearValue],
                       pipeline: &Pipeline,
                       buffer:&Buffer<V,U,S>) -> Result<CommandBuffer<StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .vertex_input(buffer)
            .draw(buffer.len() as u32, 1, 0, 0)
            .end_render_pass()
            .finish()
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
    pub fn finish(self) -> Result<CommandBuffer<StateFinished>, vk::Result> {
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
    pub fn vertex_input<V:VertexSource, U:Usage, S:SharingMode>(self, buffer: &Buffer<V,U,S>) -> Self {
        unsafe {
            self.device.inner().cmd_bind_vertex_buffers(
                self.raw,
                0,
                &[buffer.raw()],
                &[0]
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
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_command_pool(self.raw, None) }
    }
}