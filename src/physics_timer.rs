use sdl2;
use std::time::{Duration, Instant};

pub struct PhysicsTimer {
    previous: Instant,
    delta: Duration,
    steps_per_second: u64,
}

impl PhysicsTimer {
    pub fn new(steps_per_second:u64) -> Self {
        Self {
            previous: Instant::now(),
            delta: Duration::default(),
            steps_per_second
        }
    }
    pub fn update(&mut self) {
        let current = Instant::now();
        let delta = current - self.previous;
        let proportion = delta.as_millis() as u64 * self.steps_per_second;
        let current = if proportion < 1000 {
            let sleep_time = (1000 - proportion) / self.steps_per_second;
            std::thread::sleep(Duration::from_millis(sleep_time));
            Instant::now()
        }else{
            current
        };
        self.delta = current - self.previous;
        self.previous = current;
    }
    pub fn delta(&self) -> u64 {
        self.delta.as_millis() as u64
    }
    pub fn delta_f32(&self) -> f32 {
        self.delta.as_millis() as f32
    }
}
