import ImagePredictor from './module_v2.js'; 

const { createApp } = Vue;

createApp({
  components: { ImagePredictor },
  data() {
    return {
      message: 'By: Srihari | Sri Prasad | Athithyaa',
      thresholds: {}, 
    };
  },
  methods: {
  fetchThresholds() {
    fetch('/get-thresholds/')
      .then(response => response.json())
      .then(data => {
        this.thresholds = data;
      })
      .catch(error => {
        console.error('Error fetching thresholds:', error);
      });
    }
  },
  mounted() {
    this.fetchThresholds();
  },
  template: `
    <div>
      <header style="background: #f4f4f4; padding: 1em; border-bottom: 1px solid #ccc;">
        <h1>Defect detection</h1>
        <p style="margin: 0;">Status: Ready | Model: Quantized TFLite </p>
      </header>
      <main style="padding: 2em;">
        <div v-if="Object.keys(thresholds).length > 0">
          <h3>Optimal Thresholds</h3>
          <table border="1" cellspacing="0" cellpadding="5">
            <thead>
              <tr>
                <th>Class Name</th>
                <th>Optimal Threshold</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(value, key) in thresholds" :key="key">
                <td>{{ key }}</td>
                <td>{{ value }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <image-predictor />
        <div>
          <h3>{{ message }}</h3>
        </div>
      </main>
    </div>
  `
}).mount('#app');
