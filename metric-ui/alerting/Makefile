VERSION ?= 0.1.4
REGISTRY ?= quay.io/rh-ee-pezhao/appeng

build_ui:
	podman build --platform linux/amd64 -t vllm-alert-receiver:$(VERSION) .

build_and_push_ui: build_ui
	podman tag vllm-alert-receiver:$(VERSION) $(REGISTRY)/vllm-alert-receiver:$(VERSION)
	podman push $(REGISTRY)/vllm-alert-receiver:$(VERSION)