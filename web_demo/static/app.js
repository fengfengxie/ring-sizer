const form = document.getElementById("measureForm");
const imageInput = document.getElementById("imageInput");
const statusText = document.getElementById("statusText");
const inputPreview = document.getElementById("inputPreview");
const debugPreview = document.getElementById("debugPreview");
const inputFrame = document.getElementById("inputFrame");
const debugFrame = document.getElementById("debugFrame");
const jsonOutput = document.getElementById("jsonOutput");
const jsonLink = document.getElementById("jsonLink");
const defaultSampleUrl = window.DEFAULT_SAMPLE_URL || "";

const setStatus = (text) => {
  statusText.textContent = text;
};

const showImage = (imgEl, frameEl, url) => {
  if (!url) return;
  imgEl.src = url;
  frameEl.classList.add("show");
  frameEl.querySelector(".placeholder").style.display = "none";
};

const buildMeasureSettings = () => {
  const fingerSelect = form.querySelector('select[name="finger_index"]');
  const edgeSelect = form.querySelector('select[name="edge_method"]');
  return {
    finger_index: fingerSelect ? fingerSelect.value : "index",
    edge_method: edgeSelect ? edgeSelect.value : "auto",
  };
};

const runMeasurement = async (endpoint, formData, inputUrlFallback = "") => {
  setStatus("Measuring… Please wait.");
  jsonOutput.textContent = "{\n  \"status\": \"processing\"\n}";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      setStatus(error.error || "Measurement failed");
      return;
    }

    const data = await response.json();
    jsonOutput.textContent = JSON.stringify(data.result, null, 2);
    jsonLink.href = data.result_json_url || "#";

    showImage(inputPreview, inputFrame, data.input_image_url || inputUrlFallback);
    showImage(debugPreview, debugFrame, data.result_image_url);

    if (data.success) {
      setStatus("Measurement complete. Results updated.");
    } else {
      setStatus("Measurement failed. Check fail_reason.");
    }
  } catch (error) {
    setStatus("Network error. Please retry.");
  }
};

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    setStatus("Sample image loaded. Upload your own photo or click Start Measurement.");
    if (defaultSampleUrl) {
      showImage(inputPreview, inputFrame, defaultSampleUrl);
    }
    return;
  }
  const url = URL.createObjectURL(file);
  showImage(inputPreview, inputFrame, url);
  setStatus("Image ready. Click to start measurement.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const settings = buildMeasureSettings();
  const formData = new FormData();
  formData.append("finger_index", settings.finger_index);
  formData.append("edge_method", settings.edge_method);

  const file = imageInput.files[0];
  if (file) {
    formData.append("image", file);
    await runMeasurement("/api/measure", formData);
    return;
  }

  await runMeasurement("/api/measure-default", formData, defaultSampleUrl);
});

if (defaultSampleUrl) {
  showImage(inputPreview, inputFrame, defaultSampleUrl);
  setStatus("Sample image loaded. Upload your own photo or click Start Measurement.");
}
