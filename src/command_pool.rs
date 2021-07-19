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


pub struct StateClear {}

pub struct StateBegan {}

pub struct StateRenderPassBegan {}

pub struct StateFinished {}

pub struct CommandBuffer<'a, 'b, State> {
    raw: vk::CommandBuffer,
    pool: &'a CommandPool,
    _state: PhantomData<State>,
    _resources: PhantomData<&'b ()>,
}

impl<'a, 'b, X> CommandBuffer<'a, 'b, X> {
    fn state_transition<Y>(self) -> CommandBuffer<'a, 'b, Y> {
        let Self { raw, pool, .. } = self;
        CommandBuffer { raw, pool, _state: PhantomData, _resources: PhantomData }
    }
}

impl<'a, 'b> CommandBuffer<'a, 'b, StateClear> {
    pub fn begin(self, usage: vk::CommandBufferUsageFlags) -> Result<CommandBuffer<'a, 'b, StateBegan>, vk::Result> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(usage);
        let result = unsafe {
            self.pool.device.inner().begin_command_buffer(self.raw, &command_buffer_begin_info)
        };
        result.map(move |()| self.state_transition())
    }

    pub fn single_pass(self,
                       usage: vk::CommandBufferUsageFlags,
                       render_pass: &'b RenderPass,
                       framebuffer: &'b Framebuffer,
                       render_area: vk::Rect2D,
                       clear: &'b [ClearValue],
                       pipeline: &'b Pipeline,
                       vertex_count: u32,
                       instance_count: u32,
                       first_vertex: u32,
                       first_instance: u32) -> Result<CommandBuffer<'a, 'b, StateFinished>, vk::Result> {
        self.begin(usage)?
            .render_pass(render_pass, framebuffer, render_area, clear)
            .bind_pipeline(pipeline)
            .draw(vertex_count, instance_count, first_vertex, first_instance)
            .end_render_pass()
            .finish()
    }
}

impl<'a, 'b> CommandBuffer<'a, 'b, StateBegan> {
    pub fn render_pass(self, render_pass: &'b RenderPass, framebuffer: &'b Framebuffer, render_area: vk::Rect2D, clear: &'b [ClearValue]) -> CommandBuffer<'a, 'b, StateRenderPassBegan> {
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.raw())
            .framebuffer(framebuffer.raw())
            .render_area(render_area)
            .clear_values(clear);

        unsafe {
            self.pool.device.inner().cmd_begin_render_pass(
                self.raw,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            )
        }
        self.state_transition()
    }
    pub fn finish(self) -> Result<CommandBuffer<'a, 'b, StateFinished>, vk::Result> {
        let result = unsafe {
            self.pool.device.inner().end_command_buffer(
                self.raw,
            )
        };
        result.map(move |()| self.state_transition())
    }
}

impl<'a, 'b> CommandBuffer<'a, 'b, StateRenderPassBegan> {
    pub fn bind_pipeline(self, pipeline: &'b Pipeline) -> Self {
        unsafe {
            self.pool.device.inner().cmd_bind_pipeline(
                self.raw,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.raw(),
            );
        }
        self
    }
    pub fn draw(self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Self {
        unsafe {
            self.pool.device.inner().cmd_draw(
                self.raw,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
        self
    }
    pub fn end_render_pass(self) -> CommandBuffer<'a, 'b, StateBegan> {
        unsafe {
            self.pool.device.inner().cmd_end_render_pass(
                self.raw,
            );
        }
        self.state_transition()
    }
}

impl<'a, 'b> CommandBuffer<'a, 'b, StateFinished> {
    pub fn submit(&self, wait_for: &[(&Semaphore, vk::PipelineStageFlags)], then_signal: &[Semaphore], fence_to_signal:Option<&Fence>) -> VkResult<()> {
        let wait_semaphores: Vec<vk::Semaphore> = wait_for.iter().map(|(s, _)| s.raw()).collect();
        let wait_stages: Vec<vk::PipelineStageFlags> = wait_for.iter().map(|(_, s)| *s).collect();
        let signal_semaphores: Vec<vk::Semaphore> = then_signal.iter().map(Semaphore::raw).collect();
        let submit_infos = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores.as_slice())
            .signal_semaphores(signal_semaphores.as_slice())
            .command_buffers(std::slice::from_ref(&self.raw))
            .wait_dst_stage_mask(wait_stages.as_slice());
        unsafe {
            self.pool.device.inner()
                .queue_submit(
                    self.pool.device.raw_queue(),
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
        self.create_command_buffers(1).map(|v|v.into_iter().next().unwrap())
    }
    pub fn create_command_buffers(&self, count: u32) -> Result<Vec<CommandBuffer<StateClear>>, vk::Result> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.raw)
            .command_buffer_count(count)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            self.device.inner()
                .allocate_command_buffers(&command_buffer_allocate_info)
        }.map(|vec| vec.into_iter().map(|raw| CommandBuffer { raw, pool: self, _state: PhantomData, _resources: PhantomData }).collect())
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { self.device.inner().destroy_command_pool(self.raw, None) }
    }
}