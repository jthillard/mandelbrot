//! Native background worker for reference-orbit computation.
//!
//! At deep zoom the high-precision reference can take many milliseconds (tens of
//! thousands of `FBig` iterations), which would stutter the UI if done inline.
//! This runs it on a thread and coalesces bursts of requests (e.g. during a
//! drag) down to the most recent one. On the web we compute inline instead
//! (browsers need a Web Worker for threads); see `app.rs`.

use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread;

use crate::fractal::{FractalKind, compute_reference, compute_set_reference};
use crate::view::{Big, big_from_f64};

pub struct RefRequest {
    pub center_re: Big,
    pub center_im: Big,
    pub half_height: f64,
    pub julia: bool,
    pub julia_c: (f64, f64),
    pub max_iter: u32,
    pub precision: usize,
    pub kind: FractalKind,
    pub power: u32,
    /// Distortion constant for the Phoenix map (ignored by other kinds).
    pub phoenix_p: (f64, f64),
    /// Distortion constant for the Lambda map (ignored by other kinds).
    pub lambda_l: (f64, f64),
    /// Complex exponent for the Complex Multibrot kind (ignored by other kinds).
    pub complex_power: (f64, f64),
}

pub struct RefResult {
    pub center_re: Big,
    pub center_im: Big,
    pub half_height: f64,
    pub points: Vec<[f32; 2]>,
}

pub struct RefWorker {
    req_tx: Sender<RefRequest>,
    res_rx: Receiver<RefResult>,
}

impl RefWorker {
    pub fn spawn() -> Self {
        let (req_tx, req_rx) = channel::<RefRequest>();
        let (res_tx, res_rx) = channel::<RefResult>();

        thread::Builder::new()
            .name("reference-orbit".into())
            .spawn(move || {
                while let Ok(mut req) = req_rx.recv() {
                    // Coalesce: if newer requests are already queued, skip to the
                    // latest so a fast drag doesn't compute every intermediate view.
                    loop {
                        match req_rx.try_recv() {
                            Ok(newer) => req = newer,
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }

                    let points = compute(&req);
                    if res_tx
                        .send(RefResult {
                            center_re: req.center_re,
                            center_im: req.center_im,
                            half_height: req.half_height,
                            points,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })
            .expect("spawn reference-orbit thread");

        Self { req_tx, res_rx }
    }

    pub fn request(&self, req: RefRequest) {
        let _ = self.req_tx.send(req);
    }

    /// Drain all pending results, returning only the most recent.
    pub fn try_take_latest(&self) -> Option<RefResult> {
        let mut latest = None;
        while let Ok(res) = self.res_rx.try_recv() {
            latest = Some(res);
        }
        latest
    }
}

fn compute(req: &RefRequest) -> Vec<[f32; 2]> {
    if req.julia {
        let jr = big_from_f64(req.julia_c.0, req.precision);
        let ji = big_from_f64(req.julia_c.1, req.precision);
        compute_reference(
            &req.center_re,
            &req.center_im,
            &jr,
            &ji,
            req.max_iter,
            req.precision,
            req.kind,
            req.power,
            req.phoenix_p,
            req.lambda_l,
            req.complex_power,
        )
    } else {
        compute_set_reference(
            &req.center_re,
            &req.center_im,
            req.max_iter,
            req.precision,
            req.kind,
            req.power,
            req.phoenix_p,
            req.lambda_l,
            req.complex_power,
        )
    }
}
