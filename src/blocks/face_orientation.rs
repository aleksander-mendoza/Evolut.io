
#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum FaceOrientation {
    YPlus = 0,
    YMinus = 1,
    XPlus = 2,
    XMinus = 3,
    ZPlus = 4,
    ZMinus = 5,
}

impl FaceOrientation {
    pub fn is_side(&self) -> bool {
        (self.clone() as u8) > 1
    }
    pub fn opposite(&self) -> FaceOrientation {
        assert_eq!(std::mem::size_of::<Self>(), std::mem::size_of::<u8>());
        let m = self.clone() as u8;
        unsafe {
            if m % 2 == 0 {
                std::mem::transmute(m + 1)
            } else {
                std::mem::transmute(m - 1)
            }
        }
    }
}
