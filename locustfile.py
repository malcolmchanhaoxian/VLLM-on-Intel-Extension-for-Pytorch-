from locust import HttpUser, task, between

class VLLMUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def vllm_completions(self):
        response = self.client.post(
            "/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "/home/dellubuntu/mylab/vllm/model_files/",
                "prompt": "Tell me Intel history",
                "max_tokens": 128,
                "temperature": 0.9
            }
        )
        print(response.text)
