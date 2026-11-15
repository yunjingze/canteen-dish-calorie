const { BASE_URL } = require("../../config");

const PLABEL = { small: "小份", normal: "常规", large: "大份" };
const PKEYS = ["small", "normal", "large"];

Page({
  data: {
    mode: "single",
    mixedEnabled: false,
    img: "",
    portion: "normal",
    grams: "",
    loading: false,
    err: "",
    top3: [],
    scopeWarn: "",
    parts: [],
    total: null,
    note: "",
    empty: false,
    emptyMsg: "",
    portions: PKEYS.map((key) => ({ key, label: PLABEL[key] })),
  },

  onLoad() {
    wx.request({
      url: BASE_URL + "/api/health",
      success: (r) => {
        if (r.data && r.data.ok) {
          this.setData({ mixedEnabled: !!r.data.mixed_enabled });
        }
      },
      fail: () => this.setData({ err: "连不上服务：检查 config.js 里的 BASE_URL，以及电脑防火墙是否放行端口" }),
    });
  },

  switchMode(e) {
    const mode = e.currentTarget.dataset.mode;
    if (mode === this.data.mode) return;
    if (mode === "mixed" && !this.data.mixedEnabled) {
      wx.showToast({ title: "混合模型未加载", icon: "none" });
      return;
    }
    this.setData({ mode, top3: [], scopeWarn: "", parts: [], total: null, note: "", err: "", empty: false, emptyMsg: "" });
    if (this.data.img) this.upload(this.data.img);
  },

  pick() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      sizeType: ["compressed"],
      success: (res) => {
        const path = res.tempFiles[0].tempFilePath;
        this.setData({ img: path, err: "" });
        this.upload(path);
      },
    });
  },

  setPortion(e) {
    const portion = e.currentTarget.dataset.key;
    if (portion === this.data.portion) return;
    this.setData({ portion });
    if (this.data.mode === "single") {
      if (this.data.img) this.upload(this.data.img);
    } else if (this.data.parts.length) {
      this.recalc();
    }
  },

  onGramsInput(e) {
    this.setData({ grams: e.detail.value });
  },

  applyGrams() {
    if (this.data.mode === "single") {
      if (this.data.img) this.upload(this.data.img);
    } else if (this.data.parts.length) {
      this.recalc();
    }
  },

  upload(path) {
    const { mode, portion, grams } = this.data;
    // portion is a query parameter on the server, not a multipart field; grams
    // is read as a form field, so the two travel by different routes.
    const formData = mode === "single" && grams ? { grams: String(grams) } : {};
    this.setData({ loading: true, err: "" });
    const url = BASE_URL + (mode === "mixed" ? "/api/mixed/predict" : "/api/predict") +
      "?portion=" + encodeURIComponent(portion);
    wx.uploadFile({
      url,
      filePath: path,
      name: "file",
      formData,
      success: (res) => {
        let data = null;
        try {
          data = JSON.parse(res.data);
        } catch (err) {
          data = null;
        }
        if (!data) {
          this.setData({ loading: false, err: "服务返回的不是 JSON：" + String(res.data).slice(0, 80) });
          return;
        }
        if (data.error) {
          this.setData({ loading: false, err: data.error });
          return;
        }
        if (mode === "mixed") this.showMixed(data);
        else this.showSingle(data);
      },
      fail: () => this.setData({ loading: false, err: "上传失败：手机和电脑要在同一 Wi-Fi，BASE_URL 要写电脑的 IP" }),
    });
  },

  showSingle(data) {
    const scope = data.scope || {};
    const top3 = (data.top3 || []).map((x) => Object.assign({}, x, { probPct: (x.prob * 100).toFixed(1) }));
    this.setData({
      loading: false,
      note: data.note || "",
      scopeWarn: scope.in_scope === false ? (scope.message || "这张图可能不在支持的 20 类里，下面只是最接近的类别。") : "",
      top3,
      parts: [],
      total: null,
      empty: !top3.length,
      emptyMsg: "每个候选的把握都低于 10%，与其硬给一个答案，不如说认不出。换一张光线更足、菜占画面更大的照片试试。",
    });
  },

  showMixed(data) {
    const plate = data.plate || {};
    const parts = (plate.parts || []).map((p) =>
      Object.assign({}, p, { probPct: (p.prob * 100).toFixed(1), portion: this.data.portion, grams: "" })
    );
    const scope = data.scope || {};
    this.setData({
      loading: false, note: data.note || "",
      scopeWarn: scope.in_scope === false
        ? (scope.message || "这张照片可能不是一盘由支持的 20 道菜组成的餐盘，下面的部分仅供参考。")
        : "",
      top3: [], parts, total: plate.total || null,
      empty: !parts.length,
      emptyMsg: "这盘菜没有一样过线：可能整盘就是一道菜（切到单菜识别更合适），也可能是清单外的菜。",
    });
  },

  setPartPortion(e) {
    const i = e.currentTarget.dataset.index;
    const portion = e.currentTarget.dataset.key;
    const patch = {};
    patch["parts[" + i + "].portion"] = portion;
    this.setData(patch, () => this.recalc());
  },

  onPartGramsInput(e) {
    const i = e.currentTarget.dataset.index;
    const patch = {};
    patch["parts[" + i + "].grams"] = e.detail.value;
    this.setData(patch);
  },

  applyPartGrams() {
    this.recalc();
  },

  recalc() {
    const parts = this.data.parts.map((p) => ({
      class: p.class,
      prob: p.prob,
      portion: p.portion || this.data.portion,
      grams: p.grams ? String(p.grams) : null,
    }));
    if (!parts.length) return;
    wx.request({
      url: BASE_URL + "/api/mixed/total",
      method: "POST",
      header: { "content-type": "application/json" },
      data: { portion: this.data.portion, parts },
      success: (r) => {
        if (!r.data || !r.data.parts) {
          this.setData({ err: "重算热量失败" });
          return;
        }
        const merged = this.data.parts.map((p, i) => {
          const row = r.data.parts[i];
          if (!row) return p;
          return Object.assign({}, p, row, { probPct: p.probPct, portion: p.portion, grams: p.grams });
        });
        this.setData({ parts: merged, total: r.data.total, err: "" });
      },
      fail: () => this.setData({ err: "连不上服务，无法重算热量" }),
    });
  },
});
