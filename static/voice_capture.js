/**
 * Enhanced Voice Recognition Script for "Capture Text" Command
 * Specifically designed to trigger OCR functionality when "capture text" is spoken
 */

class VoiceCaptureController {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.isInitialized = false;
        this.captureCommands = [
            'capture text',
            'read text', 
            'scan text',
            'capture',
            'read this',
            'scan this',
            'what do you see',
            'read it'
        ];
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        if (this.isInitialized) return;
        
        // Check for speech recognition support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported in this browser');
            return;
        }

        this.setupRecognition();
        this.isInitialized = true;
        console.log('Voice Capture Controller initialized');
    }

    setupRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        // Configure recognition settings
        this.recognition.continuous = true;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';
        this.recognition.maxAlternatives = 1;

        // Event handlers
        this.recognition.onresult = (event) => this.handleSpeechResult(event);
        this.recognition.onerror = (event) => this.handleSpeechError(event);
        this.recognition.onend = () => this.handleSpeechEnd();
        this.recognition.onstart = () => this.handleSpeechStart();
    }

    handleSpeechResult(event) {
        const lastResult = event.results[event.results.length - 1];
        if (!lastResult.isFinal) return;

        const transcript = lastResult[0].transcript.toLowerCase().trim();
        console.log('Voice input detected:', transcript);

        // Check if any capture command is detected
        const isCapture = this.captureCommands.some(cmd => 
            transcript.includes(cmd) || 
            transcript.replace(/[^\w\s]/g, '').includes(cmd.replace(/[^\w\s]/g, ''))
        );

        if (isCapture) {
            console.log('Capture text command detected, triggering OCR...');
            this.triggerTextCapture();
        }
    }

    handleSpeechError(event) {
        console.error('Speech recognition error:', event.error);
        
        // Auto-restart on certain errors
        if (event.error === 'no-speech' || event.error === 'audio-capture') {
            setTimeout(() => {
                if (this.isListening) {
                    this.startListening();
                }
            }, 1000);
        }
    }

    handleSpeechStart() {
        console.log('Voice recognition started');
    }

    handleSpeechEnd() {
        console.log('Voice recognition ended');
        
        // Auto-restart if we should still be listening
        if (this.isListening) {
            setTimeout(() => this.startListening(), 500);
        }
    }

    triggerTextCapture() {
        try {
            // First, ensure camera is enabled if not already
            if (typeof cameraConnected !== 'undefined' && !cameraConnected) {
                console.log('Camera not connected, enabling webcam first...');
                if (typeof enableWebcam === 'function') {
                    enableWebcam().then(() => {
                        setTimeout(() => this.executeReadText(), 1000);
                    }).catch(() => {
                        console.error('Failed to enable webcam');
                        this.showFeedback('Please enable your camera first', true);
                    });
                } else {
                    this.showFeedback('Camera access function not available', true);
                }
                return;
            }

            // Execute the read text function
            this.executeReadText();
            
        } catch (error) {
            console.error('Error triggering text capture:', error);
            this.showFeedback('Error triggering text capture', true);
        }
    }

    executeReadText() {
        // Try multiple approaches to trigger the OCR functionality
        
        // Method 1: Call readText function directly if available
        if (typeof readText === 'function') {
            console.log('Calling readText() function directly');
            readText();
            this.showFeedback('Reading text from camera...');
            return;
        }

        // Method 2: Click the OCR button if it exists
        const ocrButton = document.getElementById('ocrBtn');
        if (ocrButton) {
            console.log('Clicking OCR button');
            ocrButton.click();
            this.showFeedback('Reading text from camera...');
            return;
        }

        // Method 3: Look for any button with OCR-related text
        const buttons = document.querySelectorAll('button');
        for (const button of buttons) {
            const buttonText = button.textContent.toLowerCase();
            if (buttonText.includes('read text') || buttonText.includes('ocr') || buttonText.includes('scan')) {
                console.log('Found and clicking OCR-related button:', buttonText);
                button.click();
                this.showFeedback('Reading text from camera...');
                return;
            }
        }

        // If no method worked, show error
        console.error('No OCR function or button found');
        this.showFeedback('OCR functionality not available', true);
    }

    showFeedback(message, isError = false) {
        // Try to use existing status function
        if (typeof showStatus === 'function') {
            showStatus(message, isError);
        } else {
            // Fallback: console log
            console.log(isError ? 'ERROR:' : 'INFO:', message);
        }

        // Try to speak the feedback
        if (typeof speakText === 'function') {
            speakText(message);
        }
    }

    startListening() {
        if (!this.recognition || this.isListening) return;

        try {
            this.isListening = true;
            this.recognition.start();
            console.log('Started listening for "capture text" commands');
        } catch (error) {
            console.error('Error starting voice recognition:', error);
            this.isListening = false;
        }
    }

    stopListening() {
        if (!this.recognition || !this.isListening) return;

        try {
            this.isListening = false;
            this.recognition.stop();
            console.log('Stopped listening for voice commands');
        } catch (error) {
            console.error('Error stopping voice recognition:', error);
        }
    }

    toggle() {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    // Manual trigger method for testing
    testCapture() {
        console.log('Testing capture text functionality...');
        this.triggerTextCapture();
    }
}

// Create global instance
window.voiceCaptureController = new VoiceCaptureController();

// Auto-start listening when permissions are granted
window.addEventListener('permissions-granted', () => {
    setTimeout(() => {
        if (window.voiceCaptureController) {
            window.voiceCaptureController.startListening();
        }
    }, 1000);
});

// Expose global functions for easy access
window.startVoiceCapture = () => window.voiceCaptureController?.startListening();
window.stopVoiceCapture = () => window.voiceCaptureController?.stopListening();
window.toggleVoiceCapture = () => window.voiceCaptureController?.toggle();
window.testVoiceCapture = () => window.voiceCaptureController?.testCapture();

console.log('Voice Capture Script loaded successfully');
