// Toggle user profile dropdown
        document.getElementById('userIcon').addEventListener('click', function() {
            document.getElementById('profileDropdown').classList.toggle('active');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
            const profileIcon = document.getElementById('userIcon');
            const profileDropdown = document.getElementById('profileDropdown');

            if (!profileIcon.contains(event.target) && !profileDropdown.contains(event.target)) {
                profileDropdown.classList.remove('active');
            }
        });

        // Handle file input change - show file name
        document.getElementById('videoInput').addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name;
            if (fileName) {
                // You could add code here to show the file name if desired
                const videoPreview = document.getElementById('videoPreview');
                videoPreview.src = URL.createObjectURL(e.target.files[0]);
            }
        });

        function getCSRFToken() {
            let cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                let cookie = cookies[i].trim();
                if (cookie.startsWith('csrftoken=')) {
                    return cookie.split('=')[1];
                }
            }
            return '';
        }

        function processVideo() {
            let videoFile = document.getElementById('videoInput').files[0];
            if (!videoFile) {
                alert("Please select a video file!");
                return;
            }

            // Show loading indicator
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results-container').style.display = 'none';

            let formData = new FormData();
            formData.append("video", videoFile);

            fetch('/process-video/', {
                method: "POST",
                body: formData,
                headers: {
                    "X-CSRFToken": getCSRFToken()
                }
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                document.getElementById('loading').style.display = 'none';

                if (data.success) {
                    displayResults(data.result);
                } else {
                    alert("Error processing video: " + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert("Error processing video. Please try again.");
                console.error("Error:", error);
            });
        }

//        function displayResults(result) {
//            // Show results container
//            document.getElementById('results-container').style.display = 'block';
//
//            // Set prediction result
//            const predictionBox = document.getElementById('prediction-result');
//            predictionBox.innerHTML = `<i class="${result.prediction.toLowerCase() === 'fake' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle'}" style="margin-right: 12px;"></i>Video classified as: ${result.prediction.toUpperCase()}`;
//            predictionBox.className = `prediction-box ${result.prediction.toLowerCase()}`;
//
//            // Set confidence level
//            document.getElementById('confidence-level').innerHTML = `
//                <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">${result.confidence}</div>
//                <div style="color: #666; font-size: 0.9rem;">Confidence level indicates how certain the algorithm is about the classification</div>
//            `;
//
//            // Set frame stats
//            document.getElementById('frame-stats').innerHTML = `
//                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
//                    <span>Total Frames:</span>
//                    <span style="font-weight: bold;">${result.all_frames}</span>
//                </div>
//                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
//                    <span>Fake Frames:</span>
//                    <span style="font-weight: bold; color: #c62828;">${result.fake_frames} (${result.fake_percentage.toFixed(1)}%)</span>
//                </div>
//                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
//                    <span>Real Frames:</span>
//                    <span style="font-weight: bold; color: #2e7d32;">${result.real_frames} (${result.real_percentage.toFixed(1)}%)</span>
//                </div>
//                <div style="display: flex; justify-content: space-between;">
//                    <span>Frame Difference:</span>
//                    <span style="font-weight: bold;">${result.frame_difference}</span>
//                </div>
//            `;
//
//            // Set score averages
//            document.getElementById('score-averages').innerHTML = `
//                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
//                    <span>Avg Fake Frame Score:</span>
//                    <span style="font-weight: bold;">${result.avg_fake_frames.toFixed(4)}</span>
//                </div>
//                <div style="display: flex; justify-content: space-between;">
//                    <span>Avg Real Frame Score:</span>
//                    <span style="font-weight: bold;">${result.avg_real_frames.toFixed(4)}</span>
//                </div>
//            `;
//
//            // Set score difference
//            document.getElementById('score-difference').innerHTML = `
//                <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">${result.avg_difference.toFixed(4)}</div>
//                <div style="color: #666; font-size: 0.9rem;">The difference between fake and real frame scores</div>
//            `;
//
//            // Set plot image
//            document.getElementById('plot-image').src = `data:image/png;base64,${result.plot_data}`;
//
//            // Display frame thumbnails
//            const framesContainer = document.getElementById('frames-container');
//            framesContainer.innerHTML = '';
//
//            for (let i = 0; i < result.frame_paths.length; i++) {
//                const frameDiv = document.createElement('div');
//                frameDiv.className = 'frame-item';
//
//                const img = document.createElement('img');
//                img.src = `{% static ${result.frame_paths[i]} %}`;
//                img.alt = `Frame ${i+1}`;
//
//                const scoreSpan = document.createElement('span');
//                scoreSpan.className = 'frame-score';
//
//                const score = result.frame_scores[i];
//                scoreSpan.textContent = score.toFixed(2);
//                scoreSpan.style.backgroundColor = score > 0.5 ? 'rgba(255, 0, 0, 0.7)' : 'rgba(0, 128, 0, 0.7)';
//
//                frameDiv.appendChild(img);
//                frameDiv.appendChild(scoreSpan);
//                framesContainer.appendChild(frameDiv);
//            }
//        }

function displayResults(result) {
    // Show results container
    document.getElementById('results-container').style.display = 'block';

    // Set prediction result
    const predictionBox = document.getElementById('prediction-result');
    predictionBox.innerHTML = `<i class="${result.prediction.toLowerCase() === 'fake' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle'}" style="margin-right: 12px;"></i>Video classified as: ${result.prediction.toUpperCase()}`;
    predictionBox.className = `prediction-box ${result.prediction.toLowerCase()}`;

    // Set confidence level
    document.getElementById('confidence-level').innerHTML = `
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">${result.confidence}</div>
        <div style="color: #666; font-size: 0.9rem;">Confidence level indicates how certain the algorithm is about the classification</div>
    `;

    // Set frame stats
    document.getElementById('frame-stats').innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Total Frames:</span>
            <span style="font-weight: bold;">${result.all_frames}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Fake Frames:</span>
            <span style="font-weight: bold; color: #c62828;">${result.fake_frames} (${result.fake_percentage.toFixed(1)}%)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Real Frames:</span>
            <span style="font-weight: bold; color: #2e7d32;">${result.real_frames} (${result.real_percentage.toFixed(1)}%)</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Frame Difference:</span>
            <span style="font-weight: bold;">${result.frame_difference}</span>
        </div>
    `;

    // Set score averages
    document.getElementById('score-averages').innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span>Avg Fake Frame Score:</span>
            <span style="font-weight: bold;">${result.avg_fake_frames.toFixed(4)}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Avg Real Frame Score:</span>
            <span style="font-weight: bold;">${result.avg_real_frames.toFixed(4)}</span>
        </div>
    `;

    // Set score difference
    document.getElementById('score-difference').innerHTML = `
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 10px;">${result.avg_difference.toFixed(4)}</div>
        <div style="color: #666; font-size: 0.9rem;">The difference between fake and real frame scores</div>
    `;

    // Set plot image
    document.getElementById('plot-image').src = `data:image/png;base64,${result.plot_data}`;

    // Display frame thumbnails
    const framesContainer = document.getElementById('frames-container');
    framesContainer.innerHTML = '';

    // Get static path dynamically from Django
    const staticPath = "{% static 'frames/${result.frame_paths[i]}' %}"; // Ensures correct path resolution

    for (let i = 0; i < result.frame_paths.length; i++) {
        const frameDiv = document.createElement('div');
        frameDiv.className = 'frame-item';

        const img = document.createElement('img');
        img.src = staticPath + result.frame_paths[i]; // Correct static path handling
        img.alt = `Frame ${i+1}`;

        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'frame-score';

        const score = result.frame_scores[i];
        scoreSpan.textContent = score.toFixed(2);
        scoreSpan.style.backgroundColor = score > 0.5 ? 'rgba(255, 0, 0, 0.7)' : 'rgba(0, 128, 0, 0.7)';

        frameDiv.appendChild(img);
        frameDiv.appendChild(scoreSpan);
        framesContainer.appendChild(frameDiv);
    }
}


        // Add this to your existing script section
document.getElementById('videoInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // Show the preview container
        const previewContainer = document.getElementById('video-preview-container');
        previewContainer.style.display = 'block';

        // Set the video source to the selected file
        const videoPreview = document.getElementById('uploadPreview');
        videoPreview.src = URL.createObjectURL(file);

        // Display file information
        document.getElementById('video-name').textContent = file.name;

        // Format file size
        const size = file.size;
        let formattedSize;
        if (size < 1024 * 1024) {
            formattedSize = (size / 1024).toFixed(2) + ' KB';
        } else if (size < 1024 * 1024 * 1024) {
            formattedSize = (size / (1024 * 1024)).toFixed(2) + ' MB';
        } else {
            formattedSize = (size / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
        }
        document.getElementById('video-size').textContent = formattedSize;

        // Play the video after a short delay
        setTimeout(() => {
            videoPreview.play().catch(e => console.log('Auto-play prevented by browser'));
        }, 500);
    }
});