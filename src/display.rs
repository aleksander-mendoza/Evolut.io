use winit::window::Window;
use winit::error::OsError;
use crate::render::window::init_window;
use crate::render::instance::Instance;
use crate::render::device::{pick_physical_device, Device};
use crate::render::pipeline::{PipelineBuilder, Pipeline};
use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use crate::render::shader_module::ShaderModule;
use ash::vk::{ShaderStageFlags, PhysicalDevice, ClearValue};
use crate::render::command_pool::{CommandPool, CommandBuffer, StateFinished};
use crate::render::imageview::{ImageView, Color, Depth};
use crate::render::frames_in_flight::FramesInFlight;
use crate::render::surface::Surface;
use crate::render::swap_chain::SwapChain;
use crate::render::vulkan_context::VulkanContext;
use failure::Error;
use ash::vk;
use crate::render::data::{VertexClr, VertexClrTex};
use crate::render::buffer::{Buffer, StageBuffer};
use crate::render::fence::Fence;
use crate::render::uniform_buffer::UniformBuffer;
use crate::render::descriptor_pool::{DescriptorPool, DescriptorSet};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::texture::{StageTexture, Dim2D, TextureView};
use crate::render::sampler::Sampler;
use crate::render::framebuffer::Framebuffer;

struct DisplayInner {
    // The order of all fields
    // is very important, because
    // they will be dropped
    // in the exact same order
    command_buffers: Vec<CommandBuffer<StateFinished>>,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_pool: DescriptorPool,
    framebuffers: Vec<Framebuffer>,
    pipeline: Pipeline,
    clear_values: Vec<ClearValue>,
    render_pass: RenderPass,
    depth_attachment: TextureView<Dim2D,Depth>,
    cmd_pool: CommandPool,
    image_views: Vec<ImageView<Color>>,
    uniforms: Vec<UniformBuffer<glm::Mat4, 1>>,
    swapchain: SwapChain,
}

struct DisplayData {
    data: StageBuffer<VertexClrTex>,
    texture: StageTexture<Dim2D>,
    sampler: Sampler,
}

impl DisplayData {
    pub fn new(vulkan: &VulkanContext) -> Result<DisplayData, failure::Error> {
        let data: [VertexClrTex; 3] = [
            VertexClrTex {
                pos: glm::vec2(0.0, -0.5),
                clr: glm::vec3(1.0, 0.0, 0.0),
                tex: glm::vec2(0.5, 1.0),
            },
            VertexClrTex {
                pos: glm::vec2(0.5, 0.5),
                clr: glm::vec3(0.0, 1.0, 0.0),
                tex: glm::vec2(1.0, 0.0),
            },
            VertexClrTex {
                pos: glm::vec2(-0.5, 0.5),
                clr: glm::vec3(0.0, 0.0, 1.0),
                tex: glm::vec2(0.0, 0.0),
            },
        ];
        let mut cmd_pool = CommandPool::new(vulkan.device())?;
        let data = StageBuffer::new(vulkan.device(), &cmd_pool, &data)?;
        let texture = StageTexture::new(vulkan.device(), "assets/img/wall.jpg".as_ref(), &cmd_pool, true)?;
        let (_, data) = data.get()?;
        let (_, texture) = texture.get()?;
        let sampler = Sampler::new(vulkan.device(), vk::Filter::NEAREST, true)?;
        Ok(Self { texture, data, sampler })
    }
}


impl DisplayInner {
    pub fn new(vulkan: &VulkanContext, data: &DisplayData) -> Result<Self, failure::Error> {
        let swapchain = vulkan.instance().create_swapchain(vulkan.device(), vulkan.surface())?;
        let image_views = swapchain.create_image_views()?;
        let uniforms: Result<Vec<UniformBuffer<glm::Mat4, 1>>, vk::Result> = (0..swapchain.len()).into_iter().map(|_| UniformBuffer::new(vulkan.device(), glm::translation(&glm::vec3(0., 0.2, 0.2)))).collect();
        let uniforms = uniforms?;
        let descriptor_layout = DescriptorLayout::new_sampler_uniform(&data.sampler, uniforms[0].buffer())?;
        let cmd_pool = CommandPool::new(vulkan.device())?;
        let depth_attachment = TextureView::depth_buffer_for(&swapchain)?;
        let render_pass = RenderPassBuilder::new()
            .color_attachment(swapchain.format())
            .depth_attachment(&depth_attachment)
            .graphics_subpass_with_depth([], [vk::AttachmentReference::builder()
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .attachment(0)
                .build()], vk::AttachmentReference::builder()
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .attachment(1)
                .build())
            .dependency(vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
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
            .vertex_input(0, data.data.gpu())
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
        let framebuffers: Result<Vec<Framebuffer>, vk::Result> = image_views.iter().map(|v| Framebuffer::new_color_and_depth(&render_pass, &swapchain, v, depth_attachment.imageview())).collect();
        let framebuffers = framebuffers?;
        let clear_values = vec![vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }, vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.,
                stencil: 0
            },
        }];
        let descriptor_pool = DescriptorPool::new(vulkan.device(), &descriptor_layout, &swapchain)?;
        let descriptor_sets = descriptor_pool.create_sets(&std::iter::repeat(descriptor_layout).take(swapchain.len()).collect::<Vec<DescriptorLayout>>())?;
        for (ds, u) in descriptor_sets.iter().zip(uniforms.iter()) {
            ds.update_sampler(0, &data.sampler, data.texture.imageview());
            ds.update_buffer(1, u.buffer());
        }
        let command_buffers: Result<Vec<CommandBuffer<StateFinished>>, vk::Result> = cmd_pool.create_command_buffers(framebuffers.len() as u32)?
            .into_iter().zip(framebuffers.iter().zip(descriptor_sets.iter())).map(|(cmd, (fb, ds))|
            cmd.single_pass_vertex_input_uniform(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE, &render_pass, fb, ds, swapchain.render_area(), &clear_values, &pipeline, &data.data.gpu())
        ).collect();
        let command_buffers = command_buffers?;
        Ok(Self {
            depth_attachment,
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
            descriptor_sets,
        })
    }
}

pub struct Display {
    inner: DisplayInner,
    vulkan: VulkanContext,
    data: DisplayData,
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
        self.inner = DisplayInner::new(&self.vulkan, &self.data)?;
        Ok(())
    }

    pub fn new(vulkan: VulkanContext) -> Result<Display, failure::Error> {
        let data = DisplayData::new(&vulkan)?;
        let inner = DisplayInner::new(&vulkan, &data)?;
        Ok(Self {
            inner,
            vulkan,
            data,
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
        Ok(is_suboptimal || is_resized)
    }
}