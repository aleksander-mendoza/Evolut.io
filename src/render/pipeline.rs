use crate::render::shader_module::{ShaderModule, AnyShaderModule, Fragment, Vertex};
use ash::vk;
use crate::render::device::Device;
use ash::version::DeviceV1_0;
use std::ffi::{CStr, CString};
use failure::err_msg;
use crate::render::render_pass::{RenderPassBuilder, RenderPass};
use std::rc::Rc;
use ash::vk::PipelineLayout;
use crate::render::data::{VertexSource, VertexAttrib};
use crate::render::descriptor_layout::DescriptorLayout;
use crate::render::owned_buffer::{OwnedBuffer};
use std::marker::PhantomData;
use crate::render::buffer_type::BufferType;
use crate::render::buffer::Buffer;


pub struct Pipeline {
    raw: vk::Pipeline,
    layout: vk::PipelineLayout,
    render_pass: RenderPass, // Keeping this reference prevents render pass from being deallocated
    // before pipeline. While the specification says that it's not necessary and in principle RenderPass
    // could outlive pipeline, some vendors may have bugs in their implementations. It's a lot
    // safer to keep this reference just in case.
    descriptor_layout: Vec<DescriptorLayout>, // Just keeping reference
}

impl Pipeline {
    pub fn render_pass(&self) -> &RenderPass{
        &self.render_pass
    }
    pub fn device(&self) -> &Device {
        self.render_pass.device()
    }
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }
    pub fn layout(&self) -> vk::PipelineLayout{
        self.layout
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device().inner().destroy_pipeline(self.raw, None);
            self.device().inner().destroy_pipeline_layout(self.layout, None);
            // Safety: The pipeline is dropped first. Then the layout and the render pass is dropped
            // last (unless some other pipeline uses it too)
        }
    }
}

pub struct PipelineBuilder {
    viewport: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,
    push_constants: Vec<vk::PushConstantRange>,
    shaders: Vec<(String, AnyShaderModule)>,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo,
    depth_state_create_info: vk::PipelineDepthStencilStateCreateInfo,
    color_blend_attachment_states: Vec<vk::PipelineColorBlendAttachmentState>,
    color_blend_state: vk::PipelineColorBlendStateCreateInfo,
    topology: vk::PrimitiveTopology,
    vertex_input_attribute:Vec<vk::VertexInputAttributeDescription>,
    vertex_input_binding:Vec<vk::VertexInputBindingDescription>,
    descriptor_layout: Vec<DescriptorLayout>

}
#[derive(Copy,Clone,Eq, PartialEq)]
pub struct PushConstant<T:VertexAttrib>(u32,PhantomData<T>);
impl <T:VertexAttrib> PushConstant<T>{pub fn offset(&self)->u32{self.0}}
#[derive(Copy,Clone,Eq, PartialEq)]
pub struct BufferBinding<T:VertexSource>(u32,PhantomData<T>);
impl <T:VertexSource> BufferBinding<T>{pub fn binding(&self)->u32{self.0}}

impl PipelineBuilder {
    pub fn new() -> Self {
        let stencil_state = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        Self {
            viewport: vec![],
            scissors: vec![],
            push_constants: vec![],
            shaders: vec![],
            rasterizer: vk::PipelineRasterizationStateCreateInfo::builder()
                .line_width(1.0)
                .build(),
            multisample_state_create_info: vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            depth_state_create_info: vk::PipelineDepthStencilStateCreateInfo::builder()
                .front(stencil_state)
                .back(stencil_state)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .depth_compare_op(vk::CompareOp::LESS)
                .build(),
            color_blend_attachment_states: vec![],
            color_blend_state: vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op(vk::LogicOp::COPY)
                .blend_constants([0.0, 0.0, 0.0, 0.0])
                .build(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            vertex_input_attribute: vec![],
            vertex_input_binding: vec![],
            descriptor_layout: vec![]
        }
    }
    pub fn push_constant<T:VertexAttrib>(&mut self)->PushConstant<T>{
        let offset = self.push_constants.last().map(|r|r.offset+r.size).unwrap_or(0);
        self.push_constants.push(vk::PushConstantRange{
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset ,
            size: std::mem::size_of::<T>() as u32
        });
        PushConstant(offset,PhantomData)
    }
    pub fn depth_bounds(&mut self, min:f32,max:f32)->&mut Self{
        self.depth_state_create_info.depth_bounds_test_enable = vk::TRUE;
        self.depth_state_create_info.max_depth_bounds = max;
        self.depth_state_create_info.min_depth_bounds = min;
        self
    }
    pub fn stencil_test(&mut self, enable:bool)->&mut Self{
        self.depth_state_create_info.stencil_test_enable = enable.into();
        self
    }
    pub fn depth_test(&mut self, enable:bool)->&mut Self{
        self.depth_state_create_info.depth_test_enable = enable.into();
        self.depth_state_create_info.depth_write_enable = enable.into();
        self
    }
    pub fn descriptor_layout(&mut self, layout:DescriptorLayout)->&mut Self{
        self.descriptor_layout.push(layout);
        self
    }

    pub fn color_blend_attachment_states(&mut self, blend_state: vk::PipelineColorBlendAttachmentState) -> &mut Self {
        self.color_blend_attachment_states.push(blend_state);
        self
    }
    pub fn color_blend_attachment_states_disabled(&mut self) -> &mut Self{
        self.color_blend_attachment_states(vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        })
    }
    pub fn color_blend_attachment_states_default(&mut self) -> &mut Self{
        self.color_blend_attachment_states(vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        })
    }
    pub fn reset_viewports(&mut self) -> &mut Self {
        self.viewport.clear();
        self
    }
    pub fn viewports(&mut self, viewport: vk::Viewport) -> &mut Self {
        self.viewport.push(viewport);
        self
    }
    pub fn reset_scissors(&mut self) -> &mut Self {
        self.scissors.clear();
        self
    }
    pub fn scissors(&mut self, scissors: vk::Rect2D) -> &mut Self {
        self.scissors.push(scissors);
        self
    }

    pub fn cull_face(&mut self, cull_mode: vk::CullModeFlags) -> &mut Self {
        self.rasterizer.cull_mode = cull_mode;
        self
    }

    pub fn front_face_clockwise(&mut self, clockwise: bool) -> &mut Self {
        self.rasterizer.front_face = if clockwise { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE };
        self
    }

    pub fn line_width(&mut self, line_width: f32) -> &mut Self {
        self.rasterizer.line_width = line_width;
        self
    }

    pub fn polygon_mode(&mut self, polygon_mode: vk::PolygonMode) -> &mut Self {
        self.rasterizer.polygon_mode = polygon_mode;
        self
    }
    fn any_shader(&mut self, main_func: impl ToString, shader: AnyShaderModule) -> &mut Self {
        self.shaders.push((main_func.to_string(), shader));
        self
    }
    pub fn fragment_shader(&mut self, main_func: impl ToString, shader: ShaderModule<Fragment>) -> &mut Self {
        self.any_shader(main_func,unsafe{shader.into_any()})
    }
    pub fn vertex_shader(&mut self, main_func: impl ToString, shader: ShaderModule<Vertex>) -> &mut Self {
        self.any_shader(main_func,unsafe{shader.into_any()})
    }
    pub fn topology(&mut self,topology:vk::PrimitiveTopology)->&mut Self{
        self.topology = topology;
        self
    }
    pub fn vertex_input<V:VertexSource>(&mut self, binding:u32)->BufferBinding<V>{
        self.input_buffer(binding,vk::VertexInputRate::VERTEX)
    }
    pub fn instance_input<V:VertexSource>(&mut self, binding:u32)->BufferBinding<V>{
        self.input_buffer(binding,vk::VertexInputRate::INSTANCE)
    }
    pub fn vertex_input_from<V:VertexSource,T:BufferType>(&mut self, binding:u32, _buffer:&impl Buffer<V, T>) ->BufferBinding<V>{
        self.input_buffer_from(binding,_buffer,vk::VertexInputRate::VERTEX)
    }
    pub fn instance_input_from<V:VertexSource,T:BufferType>(&mut self, binding:u32, _buffer:&impl Buffer<V,T>) ->BufferBinding<V>{
        self.input_buffer_from(binding,_buffer,vk::VertexInputRate::INSTANCE)
    }
    pub fn input_buffer_from<V:VertexSource,T:BufferType>(&mut self, binding:u32, _buffer:&impl Buffer<V,T>, input_rate:vk::VertexInputRate) ->BufferBinding<V>{
        self.input_buffer(binding, input_rate)
    }
    pub fn input_buffer<V:VertexSource>(&mut self, binding:u32, input_rate:vk::VertexInputRate)->BufferBinding<V>{
        self.vertex_input_binding.push(vk::VertexInputBindingDescription {
            binding,
            stride: std::mem::size_of::<V>() as u32,
            input_rate,
        });
        for attr in V::get_attribute_descriptions(binding){
            self.vertex_input_attribute.push(attr);
        }
        BufferBinding(binding,PhantomData)
    }

    pub fn build(&mut self, render_pass: &RenderPass) -> Result<Pipeline, failure::Error> {
        let Self {
            viewport,
            scissors,
            push_constants,
            shaders,
            rasterizer,
            multisample_state_create_info,
            depth_state_create_info,
            color_blend_attachment_states,
            color_blend_state,
            topology,
            vertex_input_attribute,
            vertex_input_binding,
            descriptor_layout,
        } = self;
        let vp = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewport)
            .scissors(scissors);
        let shader_names: Vec<CString> = shaders.iter().map(|(name, _)| CString::new(name.as_bytes()).expect("Name of shader's main function contains illegal null \\0 symbol")).collect();
        let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = shader_names.iter().zip(shaders).map(|(c_name, (_, shader))| shader.to_stage_info(c_name).build()).collect();
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_input_attribute)
            .vertex_binding_descriptions(&vertex_input_binding);
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(*topology);
        color_blend_state.attachment_count = color_blend_attachment_states.len() as u32;
        color_blend_state.p_attachments = color_blend_attachment_states.as_ptr();
        let set_layouts:Vec<vk::DescriptorSetLayout> = descriptor_layout.iter().map(|s|s.raw()).collect();
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constants)
            .set_layouts(&set_layouts);
        let pipeline_layout = unsafe { render_pass.device().inner()
            .create_pipeline_layout(&pipeline_layout_create_info, None) }?;
        let p = vk::GraphicsPipelineCreateInfo::builder()
            .viewport_state(&vp)
            .stages(shader_stages.as_slice())
            .rasterization_state(rasterizer)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .color_blend_state(color_blend_state)
            .depth_stencil_state(depth_state_create_info)
            .multisample_state(multisample_state_create_info)
            .layout(pipeline_layout)
            .base_pipeline_index(-1)
            .render_pass(render_pass.raw())
            .subpass(0);
        let result = unsafe {
            render_pass.device().inner().create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&p),
                None,
            )
        };
        fn new(pipeline: Vec<vk::Pipeline>, pipeline_layout: PipelineLayout, render_pass: &RenderPass, descriptor_layout:&Vec<DescriptorLayout>) -> Pipeline {
            Pipeline {
                raw: pipeline.into_iter().next().unwrap(),
                layout: pipeline_layout,
                render_pass: render_pass.clone(),
                descriptor_layout: descriptor_layout.clone()
            }
        }
        match result {
            Ok(pipeline) => Ok(new(pipeline, pipeline_layout, render_pass, descriptor_layout)),
            Err((pipeline, err)) => {
                new(pipeline, pipeline_layout, render_pass, descriptor_layout);
                Err(err_msg(err))
            }
        }
    }
}