/**
 * CatBreedID - Application Logic (tuần 2: nối giao diện với dịch vụ FastAPI)
 * Ảnh người dùng tải lên  -> gọi POST /predict của dịch vụ (mô hình tạm).
 * Hai ảnh mẫu trong img/   -> dùng kết quả mẫu cố định để demo hai kịch bản.
 * Không cần sửa index.html: phần gọi dịch vụ đã nằm trong file này.
 */

// DOM Selector Helper
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 2) => Number(v).toFixed(d).replace('.', ',');

// ---------------------------------------------------------------------------
// Cấu hình dịch vụ
// ---------------------------------------------------------------------------
const API_URL = window.CATBREED_API_URL || 'http://127.0.0.1:8000';
const MAX_BYTES = 8 * 1024 * 1024; // khớp với giới hạn của api/main.py

/** Gửi ảnh lên dịch vụ, trả về JSON (định dạng xem api/temp_model.py). */
async function analyzeImage(file) {
    const form = new FormData();
    form.append('file', file);
    let res;
    try {
        res = await fetch(API_URL + '/predict', { method: 'POST', body: form });
    } catch (e) {
        throw new Error('Không kết nối được dịch vụ. Hãy chạy: uvicorn api.main:app --port 8000');
    }
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || ('Dịch vụ trả lỗi ' + res.status));
    }
    return res.json();
}

// ---------------------------------------------------------------------------
// SVG Icons
// ---------------------------------------------------------------------------
const CHECK = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"></path></svg>';
const WARN = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 7.5v5.5M12 16.5v.5"></path></svg>';

// Mô tả ngắn từng giống. Mới có Abyssinian; nhóm điền thêm các giống còn lại
// (giống chưa có mô tả sẽ không hiện đoạn mô tả).
const BREED_DESC = {
    'Abyssinian': 'Giống mèo lông ngắn cổ xưa với bộ lông vằn agouti đặc trưng, đôi tai to linh hoạt và dáng vẻ nhanh nhẹn, quý phái.'
};

// ---------------------------------------------------------------------------
// Dữ liệu chế độ nâng cao của 2 ảnh mẫu (5 phương pháp, giá trị mẫu)
// ---------------------------------------------------------------------------
const ADV_DATA = {
    ok: {
        summary: '4 trên 5 phương pháp chấp nhận ảnh này',
        summaryClass: 'chip',
        note: 'Khoảng cách KNN từ chối với biên độ rất nhỏ so với ngưỡng cắt, cho thấy mẫu ảnh nằm sát biên giới hạn giữa dữ liệu học và ngoại lai (near-OOD boundary). 4/5 phương pháp xác nhận ảnh là Mèo In-Distribution.',
        rows: [
            { method: 'MSP (Max Softmax Probability)', score: '0,94', threshold: '0,62', ok: true },
            { method: 'MaxLogit', score: '8,8', threshold: '5,1', ok: true },
            { method: 'Energy Score', score: '9,5', threshold: '6,0', ok: true },
            { method: 'Mahalanobis (Feature space)', score: '+1,2', threshold: '0,0', ok: true },
            { method: 'KNN Distance (k=10)', score: '−0,04', threshold: '0,0', ok: false }
        ]
    },
    no: {
        summary: '5 trên 5 phương pháp từ chối ảnh này (Out-of-Distribution)',
        summaryClass: 'chip no',
        note: 'Tất cả 5 phương pháp hậu kiểm đều ghi nhận điểm số dưới ngưỡng an toàn (τ). Mô hình xác định độ bất định cao và từ chối đưa ra kết luận phân loại giống để tránh dự đoán sai lệch.',
        rows: [
            { method: 'MSP (Max Softmax Probability)', score: '0,31', threshold: '0,62', ok: false },
            { method: 'MaxLogit', score: '3,2', threshold: '5,1', ok: false },
            { method: 'Energy Score', score: '4,1', threshold: '6,0', ok: false },
            { method: 'Mahalanobis (Feature space)', score: '−1,6', threshold: '0,0', ok: false },
            { method: 'KNN Distance (k=10)', score: '−0,85', threshold: '0,0', ok: false }
        ]
    }
};

// ---------------------------------------------------------------------------
// Kết quả mẫu cho 2 ảnh trong img/, cùng định dạng JSON với dịch vụ
// ---------------------------------------------------------------------------
const SAMPLE_NOTICE = 'Ảnh mẫu minh họa: số liệu là giá trị mẫu, không phải kết quả của mô hình thật.';
const SAMPLE_RESULTS = {
    ok: {
        status: 'accepted', top_label: 'Abyssinian', is_mock: true, sample: true, notice: SAMPLE_NOTICE,
        top5: [
            { label: 'Abyssinian', prob: 0.94 }, { label: 'Bengal', prob: 0.03 },
            { label: 'Egyptian Mau', prob: 0.02 }, { label: 'Siamese', prob: 0.01 }
        ],
        ood: { method: 'Mahalanobis (Feature space)', score: 1.2, threshold: 0.0, accept: true },
        adv: ADV_DATA.ok
    },
    no: {
        status: 'rejected', top_label: null, is_mock: true, sample: true, notice: SAMPLE_NOTICE,
        desc: 'Điểm số tin cậy dưới ngưỡng an toàn. Đối tượng trong ảnh (American Bulldog - Chó) không nằm trong phân bố 12 giống mèo của hệ thống (Out-of-Distribution). Cơ chế từ chối đã kích hoạt để bảo vệ tính chính xác thay vì phỏng đoán sai.',
        top5: [
            { label: 'Ragdoll', prob: 0.28 }, { label: 'Birman', prob: 0.24 },
            { label: 'Persian', prob: 0.18 }, { label: 'Maine Coon', prob: 0.11 }
        ],
        ood: { method: 'Mahalanobis (Feature space)', score: -1.6, threshold: 0.0, accept: false },
        adv: ADV_DATA.no
    }
};

// Application State
let imgURL = null;
let currentFile = null;     // File người dùng tải lên (null khi đang dùng ảnh mẫu)
let pendingState = 'ok';    // chỉ dùng cho ảnh mẫu
let activeState = 'ok';

/** Apply image preview background to element */
function paint(el) {
    if (!el) return;
    if (imgURL) {
        el.style.backgroundImage = 'url("' + imgURL + '")';
        el.classList.add('has');
    } else {
        el.style.backgroundImage = '';
        el.classList.remove('has');
    }
}

/** Select a sample image from the pre-provided dataset */
function selectSample(url, name, sampleState) {
    if (imgURL && imgURL.startsWith('blob:')) {
        URL.revokeObjectURL(imgURL);
    }
    imgURL = url;
    currentFile = null;
    pendingState = sampleState || (name.toLowerCase().includes('american_bulldog') ? 'no' : 'ok');

    paint($('pvPhoto'));
    $('fname').textContent = name;
    $('drop').hidden = true;
    $('prev').hidden = false;
    $('status').textContent = '';
    $('go').disabled = false;
    $('go').textContent = 'Phân tích ảnh';

    $('prev').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/** Handle new image file upload (file picker or drag-and-drop) */
function setImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    if (imgURL && imgURL.startsWith('blob:')) {
        URL.revokeObjectURL(imgURL);
    }
    imgURL = URL.createObjectURL(file);
    currentFile = file;   // kết quả sẽ do dịch vụ trả về, không đoán theo tên file nữa

    paint($('pvPhoto'));
    $('fname').textContent = file.name;
    $('drop').hidden = true;
    $('prev').hidden = false;
    $('go').textContent = 'Phân tích ảnh';
    if (file.size > MAX_BYTES) {
        $('status').textContent = 'Ảnh quá lớn (tối đa 8 MB). Hãy chọn ảnh khác.';
        $('go').disabled = true;
    } else {
        $('status').textContent = '';
        $('go').disabled = false;
    }

    $('prev').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/** Reset application state to initial upload screen */
function reset() {
    if (imgURL && imgURL.startsWith('blob:')) {
        URL.revokeObjectURL(imgURL);
    }
    imgURL = null;
    currentFile = null;
    pendingState = 'ok';
    $('result').hidden = true;
    $('home').hidden = false;
    $('prev').hidden = true;
    $('drop').hidden = false;
    $('file').value = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/** Trích xuất phương pháp OOD tối ưu nhất (Mahalanobis Feature Space) */
function getBestOOD(r) {
    if (r.ood_all && Array.isArray(r.ood_all)) {
        const maha = r.ood_all.find(m => m.method && m.method.toLowerCase().includes('mahalanobis'));
        if (maha) return maha;
    }
    return r.ood || { method: 'Mahalanobis (Feature space)', score: r.status === 'accepted' ? 1.2 : -1.6, threshold: 0.0, accept: r.status === 'accepted' };
}

/** Render kết quả: Tập trung Dự đoán, Kiểm định OOD tối ưu tích hợp bên dưới & Grad-CAM */
function render(r) {
    const isOk = r.status === 'accepted';
    activeState = isOk ? 'ok' : 'no';
    const top = (r.top5 && r.top5[0]) ? r.top5[0] : { label: 'Chưa rõ', prob: 0 };
    const pct = Math.round(top.prob * 100);
    const notice = r.notice || (r.is_mock ? 'Kết quả giả lập từ mô hình tạm, chưa phải kết quả của mô hình thật.' : '');
    const noticeHtml = notice ? '<p class="note notice-pill">' + esc(notice) + '</p>' : '';

    // Lấy thông số OOD tối ưu nhất (Mahalanobis Feature Space)
    const bestOod = getBestOOD(r);
    const scoreFormatted = (bestOod.score > 0 ? '+' : '') + fmt(bestOod.score);
    const threshFormatted = fmt(bestOod.threshold);
    const latencyFormatted = r.latency_ms ? (r.latency_ms + ' ms') : '~45 ms';

    // HTML thẻ kiểm định OOD tốt nhất tích hợp bên dưới dự đoán
    const trustCardHtml = `
        <div class="ood-trust-card ${isOk ? 'is-ok' : 'is-bad'}">
            <div class="ood-trust-head">
                <div class="ood-trust-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    </svg>
                    <span>Kiểm định ngoại lai tối ưu (${esc(bestOod.method)})</span>
                </div>
                <span class="ood-badge ${isOk ? 'badge-ok' : 'badge-bad'}">
                    ${isOk ? CHECK + 'Đạt chuẩn In-Distribution' : WARN + 'Cảnh báo Out-of-Distribution'}
                </span>
            </div>
            <div class="ood-metrics-grid">
                <div class="ood-metric-item">
                    <span class="m-lbl">Điểm đo lường</span>
                    <span class="m-val ${isOk ? 'txt-ok' : 'txt-bad'}">${scoreFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Ngưỡng an toàn (τ)</span>
                    <span class="m-val">${threshFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Thời gian suy luận</span>
                    <span class="m-val">${latencyFormatted}</span>
                </div>
                <div class="ood-metric-item">
                    <span class="m-lbl">Trạng thái lọc</span>
                    <span class="m-val ${isOk ? 'txt-ok' : 'txt-bad'}">${isOk ? 'Cho phép nhận diện' : 'Kích hoạt từ chối'}</span>
                </div>
            </div>
            <p class="ood-trust-desc">
                ${isOk
                    ? 'Đặc trưng nơ-ron nằm an toàn trong phân phối 12 giống mèo đã học (In-Distribution). Kết quả nhận diện đạt độ tin cậy cao.'
                    : 'Đặc trưng ảnh nằm ngoài phân phối 12 giống mèo đã học (Near-OOD). Hệ thống chủ động từ chối để tránh phỏng đoán sai lệch.'}
            </p>
        </div>
    `;

    let html;
    if (isOk) {
        const desc = BREED_DESC[r.top_label];
        html = '<div class="res-head">' +
            '<span class="res-eyebrow">Kết quả phân loại giống</span>' +
            '<div class="res-title-row">' +
                '<h1>' + esc(r.top_label) + '</h1>' +
                '<span class="chip">' + CHECK + 'Khá chắc chắn</span>' +
            '</div>' +
            '<div class="meter-wrap">' +
                '<div class="meter-info"><span>Độ tự tin Softmax</span><b>' + pct + '%</b></div>' +
                '<div class="meter"><div class="mt"><i style="width:0%"></i></div></div>' +
            '</div>' +
            (desc ? '<p class="desc">' + esc(desc) + '</p>' : '') +
            noticeHtml +
            trustCardHtml +
            '<div class="res-actions">' +
                '<button class="btn pri" id="again"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg> Chọn ảnh khác</button>' +
            '</div>' +
        '</div>';
    } else {
        const desc = r.desc || 'Ảnh này khác với những ảnh mèo hệ thống đã học. Có thể trong ảnh không có mèo, hoặc đây là vật nuôi thuộc loài khác.';
        html = '<div class="res-head">' +
            '<span class="res-eyebrow">Kết quả kiểm định tập mở</span>' +
            '<div class="res-title-row">' +
                '<h1>Mình chưa chắc đây là giống nào</h1>' +
                '<span class="chip no">' + WARN + 'Chưa xác định</span>' +
            '</div>' +
            '<div class="meter-wrap">' +
                '<div class="meter-info"><span>Độ tin cậy tối đa</span><b>' + pct + '%</b></div>' +
                '<div class="meter"><div class="mt"><i style="width:0%;background:#F59E0B"></i></div></div>' +
            '</div>' +
            '<p class="desc">' + esc(desc) + '</p>' +
            noticeHtml +
            trustCardHtml +
            '<div class="res-actions">' +
                '<button class="btn pri" id="again"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg> Chọn ảnh khác</button>' +
            '</div>' +
        '</div>';
    }
    $('res').innerHTML = html;

    // Trigger smooth confidence meter animation
    requestAnimationFrame(() => {
        const bar = $('res').querySelector('.mt i');
        if (bar) {
            setTimeout(() => { bar.style.width = pct + '%'; }, 60);
        }
    });

    // Các khả năng khác: ok bỏ giống đứng đầu, từ chối hiện cả giống gần nhất
    const others = isOk ? r.top5.slice(1, 5) : r.top5.slice(0, 4);
    if ($('tips')) $('tips').hidden = isOk;
    $('why').hidden = !isOk;
    $('othTitle').textContent = isOk ? 'Các khả năng khác' : 'Máy đang phân vân giữa';
    $('othNote').hidden = isOk;
    $('oth').className = 'others' + (isOk ? '' : ' dim');
    $('oth').innerHTML = others.map(o => {
        const p = Math.round(o.prob * 100);
        return '<div class="o"><span>' + esc(o.label) + '</span><div class="mt"><i style="width:' + p + '%"></i></div><b>' + p + '%</b></div>';
    }).join('');
}

/** Switch from home to result view */
function showResult(r) {
    $('home').hidden = true;
    $('result').hidden = false;
    paint($('resPhoto'));
    paint($('hm'));
    applyOv();
    render(r);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/** Update Grad-CAM heat map opacity */
function applyOv() {
    $('ov').style.opacity = $('showOv').checked ? $('opa').value / 100 : 0;
}

// Event Listeners: Sample Buttons
document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const url = btn.dataset.img;
        const name = btn.dataset.name;
        const sampleState = btn.dataset.state || (name.toLowerCase().includes('american_bulldog') ? 'no' : 'ok');
        selectSample(url, name, sampleState);
    });
});

// Event Listeners: File Upload & Preview Actions
$('pick').onclick = () => $('file').click();
$('file').onchange = e => setImage(e.target.files[0]);
$('swap').onclick = () => { $('file').value = ''; $('file').click(); };
$('go').onclick = async () => {
    $('go').disabled = true;
    $('go').textContent = 'Đang phân tích…';
    $('status').textContent = currentFile
        ? 'Đang gửi ảnh đến dịch vụ và tính điểm từ chối…'
        : 'Đang trích xuất đặc trưng và tính toán điểm số từ chối OOD…';
    try {
        let result;
        if (currentFile) {
            result = await analyzeImage(currentFile);          // ảnh người dùng: gọi dịch vụ
        } else {
            await sleep(750);
            result = SAMPLE_RESULTS[pendingState];             // ảnh mẫu: kết quả mẫu
        }
        showResult(result);
    } catch (e) {
        $('status').textContent = e.message;                   // báo lỗi, không bịa kết quả
        $('go').disabled = false;
        $('go').textContent = 'Thử lại';
    }
};

// Drag and Drop Zone Handling
const drop = $('drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault();
    drop.classList.add('over');
}));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => {
    e.preventDefault();
    drop.classList.remove('over');
}));
drop.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        setImage(e.dataTransfer.files[0]);
    }
});
drop.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        $('file').click();
    }
});

// Dynamic Click Delegate for #again button
document.addEventListener('click', e => {
    if (e.target.id === 'again') reset();
});

// Grad-CAM Controls
$('showOv').onchange = applyOv;
$('opa').oninput = () => {
    applyOv();
    const opaVal = $('opaVal');
    if (opaVal) opaVal.textContent = $('opa').value + '%';
};

// URL Query Param Support for Testing & Demos (?demo=ok hoặc ?demo=no)
const params = new URLSearchParams(window.location.search);
if (params.get('demo') === 'ok' || params.get('mode') === 'ok') {
    imgURL = 'img/Abyssinian_1.jpg';
    showResult(SAMPLE_RESULTS.ok);
} else if (params.get('demo') === 'no' || params.get('mode') === 'no') {
    imgURL = 'img/american_bulldog_2.jpg';
    showResult(SAMPLE_RESULTS.no);
}
