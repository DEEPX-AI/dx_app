import pytest
from framework.base_test import BaseTestFramework
from framework.cli_test import CLITestFramework
from framework.config import EFFICIENTNET_CONFIG
from framework.e2e_test import E2ETestFramework
from framework.groups_test import GroupsTestFramework
from framework.integration_test import IntegrationTestFramework

pytestmark = pytest.mark.efficientnet


@pytest.mark.unit
class TestEfficientNetBase:

    @classmethod
    def setup_class(cls):
        cls.framework = BaseTestFramework(EFFICIENTNET_CONFIG)

    @pytest.mark.parametrize("script_name", EFFICIENTNET_CONFIG.variants)
    def test_preprocess(self, script_name):
        self.framework.test_preprocess(script_name)

    @pytest.mark.parametrize(
        "script_name",
        [s for s in EFFICIENTNET_CONFIG.variants if "cpp_postprocess" not in s],
    )
    def test_postprocess(self, script_name):
        self.framework.test_postprocess(script_name)


@pytest.mark.unit
class TestEfficientNetGroups:

    @classmethod
    def setup_class(cls):
        cls.framework = GroupsTestFramework(EFFICIENTNET_CONFIG)

    @pytest.mark.parametrize(
        "script_name",
        [s for s in EFFICIENTNET_CONFIG.variants if "cpp_postprocess" not in s],
    )
    def test_python_postprocess_has_method(self, script_name):
        self.framework.test_python_postprocess_has_method(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in EFFICIENTNET_CONFIG.variants if "_sync" in s]
    )
    def test_sync_has_image_inference(self, script_name):
        self.framework.test_sync_has_image_inference(script_name)


@pytest.mark.integration
class TestEfficientNetIntegration:

    @classmethod
    def setup_class(cls):
        cls.framework = IntegrationTestFramework(EFFICIENTNET_CONFIG)

    @pytest.mark.parametrize(
        "script_name", [s for s in EFFICIENTNET_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_file_not_found(self, script_name):
        self.framework.test_image_inference_file_not_found(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in EFFICIENTNET_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_success(self, script_name):
        self.framework.test_image_inference_success(script_name)


@pytest.mark.cli
class TestEfficientNetCLI:

    @classmethod
    def setup_class(cls):
        cls.framework = CLITestFramework(EFFICIENTNET_CONFIG)

    @pytest.mark.parametrize("script_name", EFFICIENTNET_CONFIG.variants)
    def test_cli_help(self, script_name):
        self.framework.test_cli_help(script_name)

    @pytest.mark.parametrize("script_name", EFFICIENTNET_CONFIG.variants)
    def test_cli_missing_required_args(self, script_name):
        self.framework.test_cli_missing_required_args(script_name)

    @pytest.mark.parametrize("script_name", EFFICIENTNET_CONFIG.variants)
    def test_cli_unrecognized_argument(self, script_name):
        self.framework.test_cli_unrecognized_argument(script_name)

    @pytest.mark.parametrize(
        "script_name", [s for s in EFFICIENTNET_CONFIG.variants if "_sync" in s]
    )
    def test_cli_image_mode(self, script_name):
        self.framework.test_cli_image_mode(script_name)


@pytest.mark.e2e
class TestEfficientNetE2E:

    @classmethod
    def setup_class(cls):
        cls.framework = E2ETestFramework(EFFICIENTNET_CONFIG)

    @pytest.mark.parametrize(
        "script_name", [s for s in EFFICIENTNET_CONFIG.variants if "_sync" in s]
    )
    def test_image_inference_real(self, script_name):
        self.framework.test_image_inference_real(script_name)
