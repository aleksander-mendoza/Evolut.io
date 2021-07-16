use std::time::{Duration, Instant};

pub struct FpsCounter {
    previous: Instant,
    delta: u64,
    frames_per_second: u64,
}

impl FpsCounter {
    pub fn new(frames_per_second:u64) -> Self {
        let previous = std::time::Instant::now();
        Self {
            previous,
            delta: 0,
            frames_per_second
        }
    }
    pub fn update(&mut self) {
        let current = std::time::Instant::now();
        let delta = (current - self.previous).as_millis() as u64;
        let proportion = delta*self.frames_per_second;
        let current = if proportion < 1000u64 {
            let sleep_time = (1000u64 - proportion) / self.frames_per_second;
            std::thread::sleep(Duration::from_millis(sleep_time as u64));
            std::time::Instant::now()
        }else{
            current
        };
        self.delta = (current - self.previous).as_millis() as u64;
        self.previous = current;
    }
    pub fn delta(&self) -> u64 {
        self.delta
    }
    pub fn delta_f32(&self) -> f32 {
        self.delta as f32
    }
    pub fn time(&self) -> Instant {
        self.previous
    }
}
