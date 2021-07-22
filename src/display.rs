
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
use crate::data::Vertex;
use crate::buffer::{Buffer, PairedBuffers};
use crate::fence::Fence;
use crate::uniform_buffer::UniformBuffer;
use crate::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::descriptor_layout::DescriptorLayout;

struct DisplayInner{
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order

    command_buffers: Vec<CommandBuffer<StateFinished>>,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_pool: DescriptorPool,
    framebuffers: Vec<Framebuffer>,
    pipeline: Pipeline,
    clear_values: [ClearValue; 1],
    render_pass: RenderPass,
    cmd_pool: CommandPool,
    image_views: Vec<ImageView>,
    uniforms:Vec<UniformBuffer<glm::Mat4,1>>,
    swapchain: SwapChain,
}

struct DisplayData{
    data:PairedBuffers<Vertex>,
    cmd_pool:CommandPool,
}

impl DisplayData{
    pub fn new(vulkan:&VulkanContext) -> Result<DisplayData, ash::vk::Result> {
        let data: [Vertex; 3] = [
            Vertex {
                pos: glm::vec2(0.0, -0.5),
                color: glm::vec3(1.0, 0.0, 0.0),
            },
            Vertex {
                pos: glm::vec2(0.5, 0.5),
                color: glm::vec3(0.0, 1.0, 0.0),
            },
            Vertex {
                pos: glm::vec2(-0.5, 0.5),
                color: glm::vec3(0.0, 0.0, 1.0),
            },
        ];
        let mut cmd_pool = CommandPool::new(vulkan.device())?;
        let fence = Fence::new(vulkan.device(), false)?;
        let future = PairedBuffers::new(vulkan.device(),&cmd_pool, fence,&data)?;
        let (_, data) = future.get()?;
        cmd_pool.clear();

        Ok(Self{data,cmd_pool})

    }
}


impl DisplayInner{
    pub fn new(vulkan: &VulkanContext, data:&DisplayData) -> Result<Self, failure::Error> {
        let swapchain = vulkan.instance().create_swapchain(vulkan.device(), vulkan.surface())?;
        let image_views = swapchain.create_image_views()?;
        let uniforms:Result<Vec<UniformBuffer<glm::Mat4,1>>,vk::Result> = (0..swapchain.len()).into_iter().map(|_|UniformBuffer::new(vulkan.device(),glm::translation(&glm::vec3(0.,0.2, 0.2)))).collect();
        let uniforms = uniforms?;
        let descriptor_layout = uniforms[0].create_descriptor_layout(0)?;
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
            .descriptor_layout(descriptor_layout.clone())
            .shader("main", frag)
            .shader("main", vert)
            .scissors(swapchain.render_area())
            .viewports(swapchain.viewport())
            .vertex_input::<Vertex>(0)
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
        let descriptor_pool = DescriptorPool::new(vulkan.device(),&swapchain)?;
        let descriptor_sets = descriptor_pool.create_sets(&std::iter::repeat(descriptor_layout).take(swapchain.len()).collect::<Vec<DescriptorLayout>>())?;
        for (ds,u) in descriptor_sets.iter().zip(uniforms.iter()){
            ds.update(u.buffer());
        }
        let command_buffers:Result<Vec<CommandBuffer<StateFinished>>,vk::Result> = cmd_pool.create_command_buffers(framebuffers.len() as u32)?
            .into_iter().zip(framebuffers.iter().zip(descriptor_sets.iter())).map(|(cmd,(fb,ds))|
            cmd.single_pass_vertex_input_uniform(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,&render_pass,fb,ds,swapchain.render_area(),&clear_values,&pipeline,&data.data.gpu())
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
            uniforms,
            clear_values,
            descriptor_pool,
            descriptor_sets
        })
    }
}

pub struct Display {
    inner: DisplayInner,
    vulkan: VulkanContext,
    data: DisplayData
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
        self.inner = DisplayInner::new(&self.vulkan,&self.data)?;
        Ok(())
    }

    pub fn new(vulkan:VulkanContext) -> Result<Display, failure::Error> {
        let data =  DisplayData::new(&vulkan)?;
        let inner = DisplayInner::new(&vulkan, &data)?;
        Ok(Self{
            inner,
            vulkan,
            data
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