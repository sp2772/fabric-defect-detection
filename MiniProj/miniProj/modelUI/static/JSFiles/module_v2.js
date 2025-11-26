export default {
  name: 'ImagePredictor',
  data() {
    return {
      imageFile: null,
      prediction: null,
      inferenceTime: null,
      imagePreviewUrl: null,
      loading: false,
      error: null,
    };
  },
  methods: {
    handleFileChange(event) {
      this.imageFile = event.target.files[0];
      this.prediction = null;
      this.inferenceTime = null;
      this.error = null;

      if (this.imageFile) {
        this.imagePreviewUrl = URL.createObjectURL(this.imageFile);
      } else {
        this.imagePreviewUrl = null;
      }
    },
    async handleSubmit() {
      if (!this.imageFile) return;

      this.loading = true;
      this.prediction = null;
      this.inferenceTime = null;
      this.error = null;

      const formData = new FormData();
      formData.append('image', this.imageFile);

      try {
        const response = await fetch('/predict/', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Prediction failed.');
        }

        const result = await response.json();
        this.prediction = result.prediction;
        this.inferenceTime = result.inference_time;
      } catch (err) {
        this.error = err.message || 'An error occurred.';
      } finally {
        this.loading = false;
      }
    },
  },
  template: `
    <div>
      <h2>Upload Image for Prediction</h2>
      <input type="file" accept="image/*" @change="handleFileChange" />
      <button @click="handleSubmit" :disabled="loading || !imageFile">
        {{ loading ? 'Predicting...' : 'Predict' }}
      </button>

      <div v-if="imagePreviewUrl" style="margin-top: 1em;">
        <img :src="imagePreviewUrl" alt="Uploaded Image" style="max-width: 100%; max-height: 300px;" />
      </div>

      <div v-if="prediction" style="margin-top: 1em;">
        <p><strong>Prediction:</strong> {{ prediction }}</p>
      </div>
      
      <div v-if="inferenceTime" style="margin-top: 1em;">
        <p><strong>Inference Time:</strong> {{ inferenceTime.toFixed(4) }} seconds</p>
      </div>

      <div v-if="error" style="color: red; margin-top: 1em;">
        {{ error }}
      </div>
    </div>
  `
};
