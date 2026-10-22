import { Link } from "react-router-dom";
import { ArrowLeft, Compass } from "lucide-react";
import { Button } from "@/components/ui/button";
import EmptyState from "@/components/EmptyState";

const NotFound = () => (
  <div className="container flex min-h-[70vh] items-center justify-center py-12">
    <EmptyState
      icon={Compass}
      title="Page not found"
      description="That route does not exist. The dashboard and upload pages are a good place to restart."
      action={
        <Button asChild>
          <Link to="/">
            <ArrowLeft /> Back to dashboard
          </Link>
        </Button>
      }
    />
  </div>
);

export default NotFound;
