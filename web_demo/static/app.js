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
const failReasonMessageMap = {
  card_not_detected:
    "Credit card not detected. Place a full card flat beside your hand.",
  hand_not_detected:
    "Hand not detected. Include your full palm in frame and keep fingers fully visible.",
  finger_isolation_failed:
    "Could not isolate the selected finger. Keep one target finger extended and separated.",
  finger_mask_too_small:
    "Finger region is too small. Move closer and use a higher-resolution photo.",
  contour_extraction_failed:
    "Finger contour extraction failed. Improve lighting and reduce background clutter.",
  axis_estimation_failed:
    "Finger axis estimation failed. Keep the finger straight and fully visible.",
  zone_localization_failed:
    "Ring zone localization failed. Keep more of the finger base visible.",
  width_measurement_failed:
    "Width measurement failed. Retake with phone parallel to the table and steady focus.",
  sobel_edge_refinement_failed:
    "Edge refinement failed. Turn on flash or use stronger, even lighting.",
  width_unreasonable:
    "Measured width is out of range. Retake with the phone parallel to the table.",
  disagreement_with_contour:
    "Edge methods disagree too much. Retake with cleaner edges and more even lighting.",
};

const formatFailReasonStatus = (failReason) => {
  if (!failReason) {
    return "Measurement failed.";
  }

  if (failReason.startsWith("quality_score_low_")) {
    return `Low edge quality detected. Turn on flash and retake. (${failReason})`;
  }

  if (failReason.startsWith("consistency_low_")) {
    return `Edge detection was inconsistent. Keep phone parallel to table and retry. (${failReason})`;
  }

  const friendlyMessage = failReasonMessageMap[failReason];
  if (friendlyMessage) {
    return `${friendlyMessage} (${failReason})`;
  }

  return `Measurement failed: ${failReason}`;
};

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
  return {
    finger_index: fingerSelect ? fingerSelect.value : "index",
    edge_method: "sobel",
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
      const failReason = data?.result?.fail_reason;
      setStatus(formatFailReasonStatus(failReason));
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
