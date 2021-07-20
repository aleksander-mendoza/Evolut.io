#[macro_use]
extern crate vk_shader_macros;

mod window;
mod instance;
mod constants;
mod platforms;
mod validation_layer;
mod device;
mod surface;
mod swap_chain;
mod block_world;
mod shader_module;
mod imageview;
mod pipeline;
mod render_pass;
mod command_pool;
mod semaphore;
mod fence;
mod frames_in_flight;
mod vulkan_context;

use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
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


struct Display {
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order
    framebuffers: Vec<Framebuffer>,
    pipeline: Pipeline,
    clear_values: [ClearValue; 1],
    render_pass: RenderPass,
    cmd_pool: CommandPool,
    image_views: Vec<ImageView>,
    swapchain: SwapChain,
    vulkan: VulkanContext
}

impl Display {
    pub fn device(&self) -> &Device {
        self.vulkan.device()
    }
    pub fn destroy(self)->VulkanContext{
        let Self{vulkan,..} = self;
        vulkan
    }

    pub fn recreate(self) -> Result<Display, Error> {
        Self::new(self.destroy())
    }

    pub fn new(vulkan: VulkanContext) -> Result<Self, failure::Error> {
        let swapchain = vulkan.instance().create_swapchain(vulkan.device(), vulkan.surface())?;
        let image_views = swapchain.create_image_views()?;

        let cmd_pool = CommandPool::new(vulkan.device())?;
        let render_pass = RenderPassBuilder::new()
            .color_attachment(swapchain.format())
            .graphics_subpass([], [ash::vk::AttachmentReference::builder()
                .layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .attachment(0)
                .build()])
            .dependency(ash::vk::SubpassDependency {
                src_subpass: ash::vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: ash::vk::AccessFlags::empty(),
                dst_access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dependency_flags: ash::vk::DependencyFlags::empty(),
            })
            .build(&vulkan.device())?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, vulkan.device())?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/blocks.vert") as &[u32], ShaderStageFlags::VERTEX, vulkan.device())?;
        let pipeline = PipelineBuilder::new()
            .shader("main", frag)
            .shader("main", vert)
            .scissors(swapchain.render_area())
            .viewports(swapchain.viewport())
            .color_blend_attachment_states(ash::vk::PipelineColorBlendAttachmentState {
                blend_enable: ash::vk::FALSE,
                color_write_mask: ash::vk::ColorComponentFlags::all(),
                src_color_blend_factor: ash::vk::BlendFactor::ONE,
                dst_color_blend_factor: ash::vk::BlendFactor::ZERO,
                color_blend_op: ash::vk::BlendOp::ADD,
                src_alpha_blend_factor: ash::vk::BlendFactor::ONE,
                dst_alpha_blend_factor: ash::vk::BlendFactor::ZERO,
                alpha_blend_op: ash::vk::BlendOp::ADD,
            })
            .build(&render_pass)?;
        let framebuffers: Result<Vec<Framebuffer>, ash::vk::Result> = image_views.iter().map(|v| v.create_framebuffer(&render_pass, &swapchain)).collect();
        let framebuffers = framebuffers?;
        let clear_values = [ash::vk::ClearValue {
            color: ash::vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        Ok(Self {
            swapchain,
            image_views,
            framebuffers,
            cmd_pool,
            render_pass,
            pipeline,
            clear_values,
            vulkan
        })
    }

    pub fn render(&mut self) -> Result<(), ash::vk::Result> {
        let fence = self.vulkan.frames_in_flight().current_fence();
        fence.wait(None)?;
        let image_available = self.vulkan.frames_in_flight().current_image_semaphore();
        let (image_idx, is_suboptimal) = self.swapchain.acquire_next_image(None, Some(image_available), None)?;
        let render_finished = self.vulkan.frames_in_flight().current_rendering();
        fence.reset()?;
        let command_buffer = self.cmd_pool.create_command_buffer()?;
        let framebuffer = &self.framebuffers[image_idx as usize];
        let command_buffers = command_buffer.single_pass(
            ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            &self.render_pass, framebuffer,
            self.swapchain.render_area(),
            &self.clear_values,
            &self.pipeline,
            3, 1, 0, 0)?;
        command_buffers.submit(&[(image_available, ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)],
                               std::slice::from_ref(render_finished),
                               Some(fence))?;
        self.swapchain.present(std::slice::from_ref(render_finished), image_idx)?;
        self.vulkan.frames_in_flight_mut().rotate();
        Ok(())
    }
}

fn main() -> Result<(), failure::Error> {
    let event_loop = EventLoop::new();
    let vulkan = VulkanContext::new(&event_loop)?;
    let mut display = Display::new(vulkan)?;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode, state, .. }, .. }, .. } => {}
            Event::LoopDestroyed => {
                unsafe {
                    display.device().device_wait_idle().expect("Failed to wait device idle!");
                }
            }
            Event::MainEventsCleared => {
                display.render().expect("Rendering failed");
            }
            _ => (),
        }
    })
}