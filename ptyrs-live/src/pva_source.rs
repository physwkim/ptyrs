//! PVA input: NTNDArray frames from an areaDetector NDCodec chain and the
//! scan position PV, parsed in the monitor callback without decoding.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use crossbeam_channel::Sender;
use epics_pva_rs::PvaResult;
use epics_pva_rs::client_native::PvaClient;
use epics_pva_rs::client_native::ops_v2::SubscriptionHandle;
use epics_pva_rs::proto::ByteOrder;
use epics_pva_rs::pv_request::PvRequestBuilder;
use epics_pva_rs::pvdata::FieldDesc;

use crate::codec::Codec;
use crate::ntnd::{FrameLayout, check_position_layout, parse_frame_head, parse_position};
use crate::pipeline::{Geometry, SourceStats, Work};
use crate::pos::{Pos, PosTable};
use crate::stats::wall_ns;

/// State carried across FRAME updates: the layout is cached per type
/// descriptor, and codec / size / timestamp persist because a monitor
/// update carries only the fields that changed.
pub struct NtndFrames {
    geom: Geometry,
    layout: Option<(usize, FrameLayout)>,
    codec: Option<Codec>,
    codec_param: Option<i32>,
    uncompressed: i64,
    ts: (i64, i32),
}

impl NtndFrames {
    pub fn new(geom: Geometry) -> Self {
        Self {
            geom,
            layout: None,
            codec: None,
            codec_param: None,
            uncompressed: 0,
            ts: (0, 0),
        }
    }

    /// One FRAME update: rejected, counted as a placeholder, or handed to
    /// [`SourceStats::accept`] as a zero-copy slice of `body`.
    pub fn on_update(
        &mut self,
        st: &mut SourceStats,
        desc: &FieldDesc,
        body: Bytes,
        order: ByteOrder,
        tx: &Sender<Work>,
    ) {
        let t_rx = wall_ns();
        let key = desc as *const FieldDesc as usize;
        if self.layout.as_ref().map(|l| l.0) != Some(key) {
            match FrameLayout::from_desc(desc) {
                Ok(l) => self.layout = Some((key, l)),
                Err(e) => return st.reject(e.to_string()),
            }
        }
        let layout = &self.layout.as_ref().unwrap().1;
        let h = match parse_frame_head(layout, &body, order) {
            Ok(h) => h,
            Err(e) => return st.reject(e.to_string()),
        };
        let (ubyte, ushort) = (layout.ubyte_sel, layout.ushort_sel);
        if let Some(c) = h.codec {
            self.codec = c;
        }
        if let Some(p) = h.codec_param {
            self.codec_param = p;
        }
        if let Some(u) = h.uncompressed_size {
            self.uncompressed = u;
        }
        if let Some((s, ns)) = h.data_time_stamp {
            self.ts = (s.unwrap_or(self.ts.0), ns.unwrap_or(self.ts.1));
        }
        let Some(uid) = h.unique_id else {
            return st.reject("update without uniqueId".into());
        };
        if uid < 0 {
            st.placeholders += 1;
            return;
        }
        let Some((sel, n, range)) = h.value else {
            return st.reject("update without value".into());
        };
        let Some(codec) = self.codec else {
            return st.reject("unknown codec.name".into());
        };
        let g = self.geom;
        let image = g.pixels() as i64;
        let ok = match codec {
            Codec::None => sel == ushort && n as i64 == image,
            _ => sel == ubyte && self.codec_param == Some(6) && self.uncompressed == 2 * image,
        };
        if !ok {
            return st.reject(format!(
                "payload does not match a {}x{} u16 image",
                g.width, g.height
            ));
        }
        let w = Work {
            payload: body.slice(range as Range<usize>),
            codec,
            uid,
            t_rx,
            t_pub: self.ts.0 * 1_000_000_000 + self.ts.1 as i64,
        };
        st.accept(w, tx);
    }
}

#[derive(Default)]
pub struct PosStats {
    pub updates: u64,
    pub rejected: u64,
    pub first_error: Option<String>,
}

/// One POS update into `table`, keyed by its uniqueId.
pub fn on_position(
    table: &PosTable,
    s: &mut PosStats,
    desc: &FieldDesc,
    body: &Bytes,
    order: ByteOrder,
) {
    match check_position_layout(desc).and_then(|()| parse_position(body, order)) {
        Ok((Some(uid), Some(x), Some(y))) if uid >= 0 => {
            let t_rx = wall_ns();
            table.insert(uid, Pos { x, y, t_rx });
            s.updates += 1;
        }
        Ok((Some(uid), ..)) if uid < 0 => {}
        Ok(_) => {
            s.rejected += 1;
            s.first_error
                .get_or_insert("partial position update".into());
        }
        Err(e) => {
            s.rejected += 1;
            s.first_error.get_or_insert(e.to_string());
        }
    }
}

/// Monitor `pv` into `table`. Give it a client, hence a TCP connection, of
/// its own, as it would have from a separate motion IOC: on the FRAME
/// connection it queued behind the frame backlog and trailed frames by up
/// to 0.9 s. The deep pipelined queue keeps the server from squashing it
/// during stalls.
pub async fn monitor_positions(
    client: &PvaClient,
    pv: &str,
    queue: u32,
    table: Arc<PosTable>,
    stats: Arc<Mutex<PosStats>>,
) -> PvaResult<SubscriptionHandle> {
    let request = PvRequestBuilder::new()
        .record("pipeline", true)
        .record("queueSize", queue)
        .build()
        .to_pv_field();
    client
        .pvmonitor_raw_frames_handle_with_request(
            pv,
            request,
            move |desc, body, order| {
                on_position(&table, &mut stats.lock().unwrap(), desc, &body, order)
            },
            |ev| eprintln!("recv POS {ev:?}"),
        )
        .await
}
