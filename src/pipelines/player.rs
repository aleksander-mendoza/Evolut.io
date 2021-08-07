use crate::blocks::Block;
use crate::input::Input;
use crate::fps::FpsCounter;
use crate::pipelines::mvp_uniforms::MvpUniforms;
use crate::pipelines::display::Display;
use crate::pipelines::game::GameResources;

pub struct Player {
    projection_matrix: glm::Mat4,
    mvp_uniforms: MvpUniforms,
    rotation: glm::Quat,
    location: glm::Vec3,
    block_in_hand: Block,
    model_matrix: glm::Mat4,
    movement_speed: f32,
    player_reach: f32,
    rotation_speed: f32,
    throw_velocity: glm::Vec3,
    ray_trace_vector: glm::Vec4,
}

impl Player {
    pub fn mvp_uniforms(&self) -> &MvpUniforms{
        &self.mvp_uniforms
    }
    pub fn location(&self) -> &glm::Vec3{
        &self.location
    }
    pub fn throw_velocity(&self) -> &glm::Vec3{
        &self.throw_velocity
    }
    pub fn new() -> Self {
        Self {
            projection_matrix: glm::identity::<f32, 4>(),
            mvp_uniforms: MvpUniforms::new(),
            rotation: glm::quat_identity(),
            location: glm::vec3(2f32, 5f32, 2f32),
            block_in_hand: Block::new(2u32),
            model_matrix: glm::identity::<f32, 4>(),
            movement_speed: 0.005f32,
            player_reach: 4f32,
            rotation_speed: 1f32,
            throw_velocity: glm::zero(),
            ray_trace_vector: glm::zero()
        }
    }


    pub fn update(&mut self, display:&mut Display<GameResources>, input: &Input, fps_counter: &FpsCounter) ->Result<(),failure::Error>{
        let ash::vk::Extent2D { width, height } = display.extent();
        let (width, height) = (width as f32, height as f32);
        if input.has_mouse_move() {
            let normalized_x = (input.mouse_move_xrel() as f32) / width
                * fps_counter.delta_f32()
                * self.rotation_speed;
            let normalized_y = (input.mouse_move_yrel() as f32) / height
                * fps_counter.delta_f32()
                * self.rotation_speed;
            self.rotation = glm::quat_angle_axis(normalized_y, &glm::vec3(1f32, 0f32, 0f32))
                * self.rotation
                * glm::quat_angle_axis(normalized_x, &glm::vec3(0f32, 1f32, 0f32));
        }

        let movement_vector = input.get_direction_unit_vector() * self.movement_speed * fps_counter.delta_f32();
        let inverse_rotation = glm::quat_inverse(&self.rotation);
        let mut movement_vector = glm::quat_rotate_vec3(&inverse_rotation, &movement_vector);
        // display.pipeline().a().world().blocks().zero_out_velocity_vector_on_hitbox_collision(&mut movement_vector, &(location-glm::vec3(0.4f32,1.5,0.4)),&(location+glm::vec3(0.4f32,0.3,0.4)));
        self.location += movement_vector;
        self.ray_trace_vector = glm::quat_rotate_vec(&inverse_rotation, &glm::vec4(0f32, 0., -self.player_reach, 0.));
        if input.has_mouse_left_click() || input.has_mouse_right_click() {
            let world = display.pipeline_mut().block_world_mut().world_mut();
            if input.has_mouse_left_click() {
                world.ray_cast_remove_block(self.location.as_slice(), self.ray_trace_vector.as_slice());
            } else {
                world.ray_cast_place_block(self.location.as_slice(), self.ray_trace_vector.as_slice(), self.block_in_hand);
            }
            world.flush_all_chunks();
            world.reset();
            display.rerecord_all_cmd_buffers()?;
        }
        if input.is_q(){
            self.throw_velocity = self.ray_trace_vector.xyz()*0.01;
        }else{
            self.throw_velocity = glm::zero();
        }
        if input.number() > -1 {
            self.block_in_hand = Block::new((input.number() + 1) as u32)
        }
        let v = glm::quat_to_mat4(&self.rotation) * glm::translation(&-self.location);

        let m = self.model_matrix;

        self.mvp_uniforms.mv = &v * m;
        self.mvp_uniforms.mvp = self.projection_matrix * &self.mvp_uniforms.mv;
        Ok(())
    }


    pub fn resize(&mut self, display:&Display<GameResources>) {
        let ash::vk::Extent2D { width, height } = display.extent();
        let fov = 60f32 / 360f32 * std::f32::consts::PI * 2f32;
        self.projection_matrix = glm::perspective(
            width as f32 / height as f32,
            fov,
            0.1f32,
            200f32,
        )
    }
}