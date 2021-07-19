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

struct Vulkan {
    entry: ash::Entry,
    window: Window,
    instance: Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: Device,
    swapchain: SwapChain,
    image_views: Vec<ImageView>,
    framebuffers: Vec<Framebuffer>,
    cmd_pool: CommandPool,
    render_pass: RenderPass,
    frames_in_flight: FramesInFlight,
    pipeline: Pipeline,
    clear_values: [ClearValue; 1],
}

impl Vulkan {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self, failure::Error> {
        let entry = unsafe { ash::Entry::new() }?;
        let window = init_window(event_loop)?;
        let instance = Instance::new(&entry, true)?;
        let surface = instance.create_surface(&entry, &window)?;
        let physical_device = instance.pick_physical_device(&surface)?;
        let device = instance.create_device(&entry, physical_device)?;
        let swapchain = instance.create_swapchain(&device, &surface)?;
        let image_views = swapchain.create_image_views()?;

        let cmd_pool = CommandPool::new(&device)?;
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
            .build(&device)?;
        let frag = ShaderModule::new(include_glsl!("assets/shaders/blocks.frag", kind: frag) as &[u32], ShaderStageFlags::FRAGMENT, &device)?;
        let vert = ShaderModule::new(include_glsl!("assets/shaders/blocks.vert") as &[u32], ShaderStageFlags::VERTEX, &device)?;
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
        let mut frames_in_flight = FramesInFlight::new(&device, 2)?;
        Ok(Self {
            entry,
            window,
            instance,
            surface,
            physical_device,
            device,
            swapchain,
            image_views,
            framebuffers,
            cmd_pool,
            render_pass,
            frames_in_flight,
            pipeline,
            clear_values,
        })
    }

    pub fn render(&mut self) -> Result<(), ash::vk::Result> {
        let fence = self.frames_in_flight.current_fence();
        fence.wait(None)?;
        let image_available = self.frames_in_flight.current_image_semaphore();
        let (image_idx, is_suboptimal) = self.swapchain.acquire_next_image(None, Some(image_available), None)?;
        let render_finished = self.frames_in_flight.current_rendering();
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
        self.swapchain.present(std::slice::from_ref(render_finished),image_idx)?;
        self.frames_in_flight.rotate();
        Ok(())
    }
}

fn main() -> Result<(), failure::Error> {
    let event_loop = EventLoop::new();
    let mut app = Vulkan::new(&event_loop)?;

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode, state, .. }, .. }, .. } => {}
            Event::LoopDestroyed => {
                unsafe {
                    app.device.device_wait_idle().expect("Failed to wait device idle!");
                }
            }
            Event::MainEventsCleared => {
                app.render().expect("Rendering failed");
            }
            _ => (),
        }
    })
}