
use winit::window::Window;
use winit::error::OsError;
use crate::window::init_window;
use crate::instance::Instance;
use crate::device::{pick_physical_device, Device};
use crate::pipeline::{PipelineBuilder, Pipeline};
use crate::render_pass::{RenderPassBuilder, RenderPass};
use crate::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PhysicalDevice, ClearValue};
use crate::command_pool::{CommandPool, CommandBuffer, StateFinished};
use crate::imageview::{Framebuffer, ImageView};
use crate::frames_in_flight::FramesInFlight;
use crate::surface::Surface;
use crate::swap_chain::SwapChain;
use crate::vulkan_context::VulkanContext;
use failure::Error;
use ash::vk;

struct DisplayInner{
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order
    command_buffers: Vec<CommandBuffer<StateFinished>>,
    framebuffers: Vec<Framebuffer>,
    pipeline: Pipeline,
    clear_values: [ClearValue; 1],
    render_pass: RenderPass,
    cmd_pool: CommandPool,
    image_views: Vec<ImageView>,
    swapchain: SwapChain,
}

impl DisplayInner{
    pub fn new(vulkan: &VulkanContext) -> Result<Self, failure::Error> {
        let swapchain = vulkan.instance().create_swapchain(vulkan.device(), vulkan.surface())?;
        let image_views = swapchain.create_image_views()?;

        let cmd_pool = CommandPool::new(vulkan.device())?;
        let render_pass = RenderPassBuilder::new()
            .color_attachment(swapchain.format())
            .graphics_subpass([], [vk::AttachmentReference::builder()
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .attachment(0)
                .build()])
            .dependency(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: vk::DependencyFlags::empty(),
            })
            .build(&vulkan.device())?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, vulkan.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/blocks.vert") as &[u32], ShaderStageFlags::VERTEX, vulkan.device())?;
        let pipeline = PipelineBuilder::new()
            .shader("main", frag)
            .shader("main", vert)
            .scissors(swapchain.render_area())
            .viewports(swapchain.viewport())
            .color_blend_attachment_states(vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                color_write_mask: vk::ColorComponentFlags::all(),
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ZERO,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
            })
            .build(&render_pass)?;
        let framebuffers: Result<Vec<Framebuffer>, vk::Result> = image_views.iter().map(|v| v.create_framebuffer(&render_pass, &swapchain)).collect();
        let framebuffers = framebuffers?;
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let command_buffers:Result<Vec<CommandBuffer<StateFinished>>,vk::Result> = cmd_pool.create_command_buffers(framebuffers.len() as u32)?
            .into_iter().zip(framebuffers.iter()).map(|(cmd,fb)|
            cmd.single_pass(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,&render_pass,fb,swapchain.render_area(),&clear_values,&pipeline,3,1,0,0)
        ).collect();
        let command_buffers = command_buffers?;
        Ok(Self {
            command_buffers,
            swapchain,
            image_views,
            framebuffers,
            cmd_pool,
            render_pass,
            pipeline,
            clear_values,
        })
    }
}

pub struct Display {
    inner: DisplayInner,
    vulkan: VulkanContext,
}

impl Display {
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn destroy(self) -> VulkanContext {
        let Self { vulkan, .. } = self;
        vulkan
    }

    pub fn recreate(&mut self) -> Result<(), failure::Error> {
        self.inner = DisplayInner::new(&self.vulkan)?;
        Ok(())
    }

    pub fn new(vulkan:VulkanContext) -> Result<Display, failure::Error> {
        let inner = DisplayInner::new(&vulkan)?;
        Ok(Self{
            inner,
            vulkan
        })
    }


    pub fn render(&mut self) -> Result<bool, vk::Result> {
        let fence = self.vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = self.vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = self.inner.swapchain.acquire_next_image(None, Some(image_available), None)?;
        let render_finished = self.vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = &self.inner.command_buffers[image_idx as usize];
        command_buffer.submit(&[(image_available, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
                               std::slice::from_ref(render_finished),
                               Some(fence))?;
        let result = self.inner.swapchain.present(std::slice::from_ref(render_finished), image_idx);
        let is_resized = match result {
            Ok(is_suboptimal) => is_suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(vk::Result::SUBOPTIMAL_KHR) => true,
            err => err?
        };
        self.vulkan.frames_in_flight_mut().rotate();
        Ok(is_suboptimal||is_resized)
    }
}