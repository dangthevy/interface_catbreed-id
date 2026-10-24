/**
 * CatBreedID - Application Logic
 * Standard Vanilla JavaScript (ES6+)
 */

// DOM Selector Helper
const $ = id => document.getElementById(id);

// SVG Icons
const CHECK = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"></path></svg>';
const WARN = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 7.5v5.5M12 16.5v.5"></path></svg>';

// Mock Other Breeds Distributions
const OTH = { 
    ok: [
        ["Bengal", 3], 
        ["Egyptian Mau", 2], 
        ["Siamese", 1]
    ], 
    no: [
        ["Ragdoll", 28], 
        ["Birman", 24], 
        ["Persian", 18], 
        ["Maine Coon", 11]
    ] 
};

// OOD Rejection Methods Data (Advanced Mode)
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

// Application State
let imgURL = null;
let currentFile = null;
let pendingState = 'ok';
let activeState = 'ok';

/**
 * Apply image preview background to element
 */
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

/**
 * Select a sample image from the pre-provided dataset
 */
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
    
    // Highlight preview card
    $('prev').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Handle new image file upload (file picker or drag-and-drop)
 */
function setImage(file) {
    if (!file || !file.type.startsWith('image/')) return;
    if (imgURL && imgURL.startsWith('blob:')) {
        URL.revokeObjectURL(imgURL);
    }
    imgURL = URL.createObjectURL(file);
    currentFile = file;

    // Detect state based on filename
    const lowerName = file.name.toLowerCase();
    if (lowerName.includes('american_bulldog') || lowerName.includes('bulldog') || lowerName.includes('dog') || lowerName.includes('cho')) {
        pendingState = 'no';
    } else {
        pendingState = 'ok';
    }

    paint($('pvPhoto')); 
    $('fname').textContent = file.name;
    $('drop').hidden = true; 
    $('prev').hidden = false; 
    $('status').textContent = ''; 
    $('go').disabled = false; 
    $('go').textContent = 'Phân tích ảnh';

    $('prev').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Reset application state to initial upload screen
 */
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

/**
 * Render result view based on classification state ('ok' or 'no')
 */
function render(s) {
    activeState = s;
    const isOk = (s === 'ok');

    $('res').innerHTML = isOk
        ? `<p class="lead">Có vẻ đây là</p>
           <h1>Abyssinian</h1>
           <span class="chip">${CHECK}Khá chắc chắn</span>
           <div class="meter">
               <div class="mt"><i style="width:0%"></i></div>
               <b>94%</b>
           </div>
           <p class="desc">Giống mèo lông ngắn cổ xưa với bộ lông vằn agouti đặc trưng, đôi tai to linh hoạt và dáng vẻ nhanh nhẹn, quý phái.</p>
           <p class="note">Mô hình xác nhận ảnh thuộc phân bố 12 giống mèo đã học (In-Distribution).</p>
           <button class="btn pri" id="again">Chọn ảnh khác</button>`
        : `<span class="chip no">${WARN}Chưa xác định</span>
           <h1>Mình chưa chắc đây là giống nào</h1>
           <div class="meter">
               <div class="mt"><i style="width:0%;background:#F59E0B"></i></div>
               <b>28%</b>
           </div>
           <p class="desc">Điểm số tin cậy dưới ngưỡng an toàn. Đối tượng trong ảnh (American Bulldog - Chó) không nằm trong phân bố 12 giống mèo của hệ thống (Out-of-Distribution). Cơ chế từ chối đã kích hoạt để bảo vệ tính chính xác thay vì phỏng đoán sai.</p>
           <button class="btn pri" id="again">Chọn ảnh khác</button>`;
    
    // Trigger smooth confidence meter animation
    requestAnimationFrame(() => {
        const bar = $('res').querySelector('.mt i');
        if (bar) {
            setTimeout(() => { bar.style.width = isOk ? '94%' : '28%'; }, 60);
        }
    });

    $('tips').hidden = isOk; 
    $('why').hidden = !isOk;
    $('othTitle').textContent = isOk ? 'Các khả năng khác' : 'Máy đang phân vân giữa';
    $('othNote').hidden = isOk; 
    $('oth').className = 'others' + (isOk ? '' : ' dim');
    $('oth').innerHTML = OTH[s].map(r => '<div class="o"><span>' + r[0] + '</span><div class="mt"><i style="width:' + r[1] + '%"></i></div><b>' + r[1] + '%</b></div>').join('');

    // Update Advanced OOD section data dynamically
    const adv = ADV_DATA[s];
    if ($('advSummary')) {
        $('advSummary').textContent = adv.summary;
        $('advSummary').className = adv.summaryClass;
    }
    if ($('advNote')) {
        $('advNote').textContent = adv.note;
    }
    if ($('advTableBody')) {
        $('advTableBody').innerHTML = adv.rows.map(row => `
            <tr>
                <td>${row.method}</td>
                <td>${row.score}</td>
                <td>${row.threshold}</td>
                <td class="${row.ok ? 'ok' : 'bad'}">${row.ok ? 'Chấp nhận' : 'Từ chối'}</td>
            </tr>
        `).join('');
    }
}

/**
 * Switch from home to result view
 */
function showResult(s) {
    $('home').hidden = true; 
    $('result').hidden = false;
    paint($('resPhoto')); 
    paint($('hm')); 
    applyOv(); 
    render(s);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Update Grad-CAM heat map opacity
 */
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
$('go').onclick = () => { 
    $('go').disabled = true; 
    $('go').textContent = 'Đang phân tích…'; 
    $('status').textContent = 'Đang trích xuất đặc trưng và tính toán điểm số từ chối OOD…'; 
    setTimeout(() => {
        showResult(pendingState);
    }, 750); 
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
$('opa').oninput = applyOv;

// URL Query Param Support for Testing & Demos (?demo=ok hoặc ?demo=no)
const params = new URLSearchParams(window.location.search);
if (params.get('demo') === 'ok' || params.get('mode') === 'ok') {
    imgURL = 'img/Abyssinian_1.jpg';
    showResult('ok');
} else if (params.get('demo') === 'no' || params.get('mode') === 'no') {
    imgURL = 'img/american_bulldog_2.jpg';
    showResult('no');
}
