#[macro_use]
extern crate vk_shader_macros;
extern crate nalgebra_glm as glm;
#[macro_use]
extern crate memoffset;

use crate::render::vulkan_context::VulkanContext;
use crate::display::{Display};
use std::ops::{Deref, DerefMut};
use crate::input::Input;
use crate::fps::FpsCounter;
use failure::err_msg;
use crate::blocks::{World, Block};
use crate::blocks::block_properties::{BEDROCK, DIRT, GRASS, PLANK};
use ash::vk;
use crate::render::command_pool::CommandBuffer;
use crate::render::framebuffer::Framebuffer;
use crate::render::descriptor_pool::DescriptorSet;
use crate::render::pipeline::Pipeline;
use crate::render::swap_chain::SwapChain;
use crate::render::render_pass::RenderPass;
use crate::render::submitter::Submitter;
use crate::mvp_uniforms::MvpUniforms;
// use crate::triangles::Triangles;
use crate::block_world::{BlockWorld, BlockWorldResources};
use crate::render::shader_module::ShaderModule;
use ash::vk::ShaderStageFlags;
use crate::particles::ParticleResources;
use crate::joint::{Joint, JointResources};
use crate::render::compute::ComputePipelineBuilder;
use crate::game::GameResources;
use crate::player::Player;

mod block_world;
mod display;
mod render;
mod input;
mod fps;
mod blocks;
mod mvp_uniforms;
mod triangles;
mod particle;
mod particles;
mod joint;
mod game;
mod player;
mod bone;
mod constraint;
mod particle_constants;
mod bones;
mod physics;
mod foundations;


fn main() -> Result<(), failure::Error> {
    run()
}

fn run() -> Result<(),failure::Error>{
    #[cfg(target_os = "macos")] {
        sdl2::hint::set("SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK", "1");
    }
    let sdl = sdl2::init().map_err(err_msg)?;
    let video_subsystem = sdl.video().map_err(err_msg)?;
    let timer = sdl.timer().map_err(err_msg)?;
    let window = video_subsystem
        .window("Game", 900, 700)
        .vulkan()
        .resizable()
        .build()?;
    sdl.mouse().set_relative_mouse_mode(true);
    let vulkan = VulkanContext::new(window)?;
    let mut player = Player::new();
    let mut display = Display::new(vulkan,&player, GameResources::new)?;

    let event_pump = sdl.event_pump().map_err(err_msg)?;
    let mut input = Input::new(event_pump);
    let mut fps_counter = FpsCounter::new(timer, 60);


    std::thread::spawn(move ||{

    });

    player.resize(&display);
    display.rerecord_all_cmd_buffers()?;
    'main: loop {
        fps_counter.update();
        input.poll();

        if input.quit() {
            break;
        }
        if input.escape() {
            input.reset_escape();
            sdl.mouse()
                .set_relative_mouse_mode(!sdl.mouse().relative_mouse_mode());
        }
        player.update(&mut display, &input, &fps_counter);
        if display.render(false,&player)? {
            display.device().device_wait_idle()?;
            display.recreate()?;
            display.rerecord_all_cmd_buffers()?;
            player.resize(&display);
        }
    }
    display.device().device_wait_idle()?;
    Ok(())
}
